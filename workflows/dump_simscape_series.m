function rows = dump_simscape_series(run_id, varargin)
% dump_simscape_series Print all available Simscape series for a run.
%
% Usage:
%   dump_simscape_series("d2090a3ceea850315fee08976ebc891d")
%   dump_simscape_series("d2090a3ceea850315fee08976ebc891d", "ModelId", "BallDrop")
%   dump_simscape_series("d2090a3ceea850315fee08976ebc891d", "UseRawSegments", true)
%   dump_simscape_series("d2090a3ceea850315fee08976ebc891d", "PrintTable", false)
%
% It reads:
%   models/simulink/<ModelId>/runs/<run_id>/debug/simlog_segments.mat
%
% and prints one row per discovered series path:
%   idx | path | points | dims | t_start | t_end

    p = inputParser;
    addRequired(p, "run_id", @(x) ischar(x) || isstring(x));
    addParameter(p, "ModelId", "BallDrop", @(x) ischar(x) || isstring(x));
    addParameter(p, "RepoRoot", "", @(x) ischar(x) || isstring(x));
    addParameter(p, "UseRawSegments", false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, "PrintTable", true, @(x) islogical(x) || isnumeric(x));
    parse(p, run_id, varargin{:});

    run_id = string(p.Results.run_id);
    model_id = string(p.Results.ModelId);
    use_raw_segments = logical(p.Results.UseRawSegments);
    print_table = logical(p.Results.PrintTable);
    repo_root = string(p.Results.RepoRoot);
    if strlength(repo_root) == 0
        repo_root = string(pwd);
    end

    mat_file = fullfile(char(repo_root), "models", "simulink", char(model_id), ...
        "runs", char(run_id), "debug", "simlog_segments.mat");
    if ~isfile(mat_file)
        error("dump_simscape_series:MissingFile", "Missing file: %s", mat_file);
    end

    S = load(mat_file);

    series_map = struct();
    if ~use_raw_segments && isfield(S, "simscapeMerged") && isstruct(S.simscapeMerged)
        series_map = S.simscapeMerged;
        mode = "simscapeMerged";
    else
        if ~isfield(S, "simlogSegments") || isempty(S.simlogSegments)
            error("dump_simscape_series:NoSegments", ...
                "No simlogSegments found in %s", mat_file);
        end
        for iseg = 1:numel(S.simlogSegments)
            seg_node = S.simlogSegments{iseg};
            seg_struct = i_simlog_node_to_struct(seg_node);
            if isempty(fieldnames(seg_struct))
                continue;
            end
            names = fieldnames(seg_struct);
            for i = 1:numel(names)
                key = names{i};
                if ~isfield(series_map, key)
                    series_map.(key) = seg_struct.(key);
                else
                    series_map.(key).Time = [series_map.(key).Time(:); seg_struct.(key).Time(:)];
                    series_map.(key).Data = [series_map.(key).Data; seg_struct.(key).Data];
                end
            end
        end
        mode = "simlogSegments(raw)";
    end

    names = sort(fieldnames(series_map));
    if print_table
        fprintf("Run: %s | Model: %s | Source: %s | Series: %d\n", ...
            run_id, model_id, mode, numel(names));
        fprintf("%4s | %-90s | %8s | %4s | %12s | %12s\n", ...
            "idx", "path", "points", "dims", "t_start", "t_end");
        fprintf("%s\n", repmat("-", 1, 150));
    end

    rows = struct("path", {}, "points", {}, "dims", {}, "t_start", {}, "t_end", {});
    for i = 1:numel(names)
        key = names{i};
        t = [];
        v = [];
        if isfield(series_map.(key), "Time")
            t = series_map.(key).Time;
        end
        if isfield(series_map.(key), "Data")
            v = series_map.(key).Data;
        end
        t = t(:);
        n = numel(t);
        if isempty(v)
            d = 0;
        elseif isvector(v)
            d = 1;
        else
            d = size(v, 2);
        end
        if n > 0
            t0 = t(1);
            t1 = t(end);
        else
            t0 = NaN;
            t1 = NaN;
        end

        if print_table
            fprintf("%4d | %-90s | %8d | %4d | %12.6g | %12.6g\n", i, key, n, d, t0, t1);
        end
        rows(end + 1) = struct( ... %#ok<AGROW>
            "path", key, ...
            "points", n, ...
            "dims", d, ...
            "t_start", t0, ...
            "t_end", t1);
    end
end


function sim_struct = i_simlog_node_to_struct(simlog_obj)
    sim_struct = struct();
    if isempty(simlog_obj)
        return;
    end

    stack = {simlog_obj};
    path_stack = {''};

    while ~isempty(stack)
        node = stack{1};
        node_path = path_stack{1};
        stack(1) = [];
        path_stack(1) = [];

        try
            s = node.series;
            if s.points > 0
                t = time(s);
                v = values(s);
                if isempty(node_path)
                    nm = node.id;
                else
                    nm = node_path;
                end
                fn = matlab.lang.makeValidName(strrep(nm, '.', '_'));
                sim_struct.(fn).Time = t(:);
                sim_struct.(fn).Data = v;
            end
        catch
            % Non-series nodes are expected in the tree.
        end

        f = fieldnames(node);
        for k = 1:numel(f)
            child = node.(f{k});
            if isa(child, 'simscape.logging.Node')
                if isempty(node_path)
                    new_path = f{k};
                else
                    new_path = [node_path '.' f{k}];
                end
                stack{end + 1} = child; %#ok<AGROW>
                path_stack{end + 1} = new_path; %#ok<AGROW>
            end
        end
    end
end
