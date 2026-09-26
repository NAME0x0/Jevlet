using System.Data.OleDb;
using System.IO;
using System.Text.RegularExpressions;

namespace Jevlet.App;

/// <summary>
/// Files for the open_file skill, from the Windows Search index (the same index Start uses), so
/// a search costs milliseconds and needs no crawling. Candidates become the model's options.
/// </summary>
internal static partial class FileSearch
{
    private const string Connection = "Provider=Search.CollatorDSO;Extended Properties='Application=Windows';";
    private static readonly HashSet<string> Filler =
    [
        "open", "my", "the", "a", "an", "file", "files", "document", "doc", "pdf", "find", "show", "me", "pull", "up",
        "bring", "load", "and", "it", "where", "is", "where's", "please", "can", "you", "i", "need", "spreadsheet", "sheet",
    ];
    private static readonly Dictionary<string, string> Paths = new(StringComparer.Ordinal);

    [GeneratedRegex(@"[a-z0-9]+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex WordPattern();

    /// <summary>File names matching the command's content words, most relevant first.</summary>
    public static List<string> Candidates(string command, int limit = 12)
    {
        var words = WordPattern().Matches(command.ToLowerInvariant()).Select(m => m.Value)
            .Where(w => w.Length > 1 && !Filler.Contains(w)).Distinct().Take(4).ToList();
        if (words.Count == 0)
        {
            return [];
        }
        var conditions = string.Join(" OR ", words.Select(w => $"CONTAINS(System.FileName, '\"{w}*\"')"));
        var sql = $"SELECT TOP {limit * 3} System.ItemNameDisplay, System.ItemPathDisplay, System.DateModified FROM SystemIndex " +
            $"WHERE SCOPE='file:' AND System.Kind <> 'folder' AND ({conditions}) ORDER BY System.DateModified DESC";
        var found = new List<(string Name, string Path, int Score)>();
        try
        {
            using var connection = new OleDbConnection(Connection);
            connection.Open();
            using var query = new OleDbCommand(sql, connection);
            using var reader = query.ExecuteReader();
            while (reader.Read())
            {
                var name = reader.GetValue(0) as string ?? "";
                var path = reader.GetValue(1) as string ?? "";
                if (name.Length == 0 || path.Contains(@"\AppData\", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                var lowered = name.ToLowerInvariant();
                found.Add((name, path, words.Count(w => lowered.Contains(w, StringComparison.Ordinal))));
            }
        }
        catch (Exception error) when (error is OleDbException or InvalidOperationException)
        {
            Log.Error("file search failed", error);
            return [];
        }
        var names = new List<string>();
        foreach (var (name, path, _) in found.OrderByDescending(f => f.Score))
        {
            if (!Paths.ContainsKey(name))
            {
                Paths[name] = path;
            }
            if (!names.Contains(name, StringComparer.Ordinal))
            {
                names.Add(name);
            }
            if (names.Count == limit)
            {
                break;
            }
        }
        return names;
    }

    public static string? PathOf(string name) => Paths.TryGetValue(name, out var path) && File.Exists(path) ? path : null;
}
