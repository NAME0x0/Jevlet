using System.Globalization;
using System.IO;

namespace Jevlet.App;

/// <summary>Append-only log at %LOCALAPPDATA%\Jevlet\logs\jevlet.log (rotated at 2 MB).</summary>
internal static class Log
{
    private static readonly Lock Gate = new();
    public static readonly string Folder = Path.Combine(AppPaths.Root, "logs");
    private static readonly string File = Path.Combine(Folder, "jevlet.log");

    public static void Info(string message) => Write("info", message);

    public static void Error(string message, Exception? error = null) =>
        Write("error", error is null ? message : $"{message}: {error}");

    private static void Write(string level, string message)
    {
        lock (Gate)
        {
            try
            {
                Directory.CreateDirectory(Folder);
                var info = new FileInfo(File);
                if (info.Exists && info.Length > 2 * 1024 * 1024)
                {
                    System.IO.File.Move(File, File + ".1", overwrite: true);
                }
                System.IO.File.AppendAllText(File,
                    $"{DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture)} {level} {message}{Environment.NewLine}");
            }
            catch (IOException)
            {
                // Logging must never take the app down.
            }
        }
    }
}

internal static class AppPaths
{
    public static readonly string Root = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Jevlet");
}
