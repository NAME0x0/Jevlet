using System.IO;
using System.Text.Json;
using Jevlet.Core;

namespace Jevlet.App;

/// <summary>User settings, stored as JSON next to the database.</summary>
internal sealed class Settings
{
    private static readonly string FilePath = Path.Combine(AppPaths.Root, "settings.json");
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    /// <summary>Palette hotkey, for example "Alt+Space" or "Ctrl+Alt+J".</summary>
    public string Hotkey { get; set; } = "Alt+Space";

    /// <summary>Folder with model.onnx, model.json, vocab.txt; empty means the default.</summary>
    public string ModelDirectory { get; set; } = "";

    public bool PreferGpu { get; set; } = true;
    public bool StartWithWindows { get; set; } = true;

    /// <summary>City used for weather when a command names none.</summary>
    public string HomeCity { get; set; } = "";

    /// <summary>"System", "Dark", or "Light".</summary>
    public string Theme { get; set; } = "System";

    public static Settings Load()
    {
        try
        {
            if (File.Exists(FilePath))
            {
                return JsonSerializer.Deserialize<Settings>(File.ReadAllText(FilePath)) ?? new Settings();
            }
        }
        catch (Exception error) when (error is IOException or JsonException)
        {
            Log.Error("settings unreadable; using defaults", error);
        }
        return new Settings();
    }

    public void Save()
    {
        Directory.CreateDirectory(AppPaths.Root);
        var temporary = FilePath + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(this, Json));
        File.Move(temporary, FilePath, overwrite: true);
    }

    /// <summary>The model folder: the setting, the installed model, or a development export.</summary>
    public string? ResolveModelDirectory()
    {
        static bool Complete(string folder) =>
            File.Exists(Path.Combine(folder, "model.onnx")) && File.Exists(Path.Combine(folder, "model.json"));

        if (ModelDirectory.Length > 0 && Complete(ModelDirectory))
        {
            return ModelDirectory;
        }
        var installed = Path.Combine(AppPaths.Root, "models", "current");
        if (Complete(installed))
        {
            return installed;
        }
        for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder is not null; folder = folder.Parent)
        {
            var exports = new DirectoryInfo(Path.Combine(folder.FullName, "data", "models", "onnx"));
            if (exports.Exists)
            {
                // Development exports: the newest one trained for this app's catalogue. A configured
                // or installed model is returned as is, and LoadModel refuses it if it does not match.
                var newest = exports.GetDirectories()
                    .Where(d => Complete(d.FullName) && !d.Name.Contains("int8", StringComparison.Ordinal)
                        && ModelCompatibility.Serves(d.FullName, Catalogue.Shared))
                    .OrderByDescending(d => d.Name, StringComparer.Ordinal)
                    .FirstOrDefault();
                if (newest is not null)
                {
                    return newest.FullName;
                }
            }
        }
        return null;
    }
}
