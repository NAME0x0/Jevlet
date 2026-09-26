using System.Diagnostics;
using System.IO;
using System.Text.Json;
using Jevlet.Core;

namespace Jevlet.App;

internal sealed record OpenWindow(nint Handle, string Title, string Process)
{
    public string Label => Snapshot.WindowLabel(Process, Title);
}

internal sealed record InstalledApp(string Name, string AppId);

/// <summary>What is on this machine right now: open windows (as Alt-Tab lists them) and apps.</summary>
internal static class Desktop
{
    private static readonly string AppsCache = Path.Combine(AppPaths.Root, "apps.json");
    private static readonly string[] Noise = ["uninstall", "help", "readme", "documentation", "release notes", "license", "website"];
    private static readonly Dictionary<uint, string> ProcessNames = [];
    private static List<InstalledApp> _apps = [];

    public static IReadOnlyList<InstalledApp> Apps => _apps;

    /// <summary>Top-level windows a person would Alt-Tab to, most recent first.</summary>
    public static List<OpenWindow> Windows(ICollection<nint> exclude)
    {
        var windows = new List<OpenWindow>();
        Native.EnumWindows((hwnd, _) =>
        {
            if (exclude.Contains(hwnd) || !Native.IsWindowVisible(hwnd) || Native.GetWindow(hwnd, Native.GW_OWNER) != 0)
            {
                return true;
            }
            var style = (long)Native.GetWindowLongPtr(hwnd, Native.GWL_EXSTYLE);
            if ((style & Native.WS_EX_TOOLWINDOW) != 0 && (style & Native.WS_EX_APPWINDOW) == 0
                || (style & Native.WS_EX_NOACTIVATE) != 0)
            {
                return true;
            }
            if (Native.DwmGetWindowAttribute(hwnd, Native.DWMWA_CLOAKED, out var cloaked, sizeof(int)) == 0 && cloaked != 0)
            {
                return true; // on another virtual desktop, or a suspended app
            }
            var title = Native.WindowText(hwnd);
            if (title.Length == 0 || !Native.GetWindowRect(hwnd, out var rect)
                || rect.Right - rect.Left < 40 || rect.Bottom - rect.Top < 40)
            {
                return true;
            }
            windows.Add(new OpenWindow(hwnd, title, ProcessName(hwnd)));
            return true;
        }, 0);
        return windows;
    }

    private static string ProcessName(nint hwnd)
    {
        Native.GetWindowThreadProcessId(hwnd, out var pid);
        lock (ProcessNames)
        {
            if (ProcessNames.TryGetValue(pid, out var cached))
            {
                return cached;
            }
            string name;
            try
            {
                using var process = Process.GetProcessById((int)pid);
                name = process.ProcessName;
            }
            catch (ArgumentException)
            {
                name = "";
            }
            ProcessNames[pid] = name;
            return name;
        }
    }

    /// <summary>Start-menu apps (Win32 and Store) from the cache; refresh in the background.</summary>
    public static void LoadApps()
    {
        try
        {
            if (File.Exists(AppsCache))
            {
                _apps = JsonSerializer.Deserialize<List<InstalledApp>>(File.ReadAllText(AppsCache)) ?? [];
            }
        }
        catch (Exception error) when (error is IOException or JsonException)
        {
            Log.Error("app cache unreadable", error);
        }
        var thread = new Thread(RefreshApps) { IsBackground = true, Name = "Jevlet app list" };
        thread.SetApartmentState(ApartmentState.STA); // the Shell's COM objects expect STA
        thread.Start();
    }

    private static void RefreshApps()
    {
        try
        {
            var watch = Stopwatch.StartNew();
            var shellType = Type.GetTypeFromProgID("Shell.Application")!;
            dynamic shell = Activator.CreateInstance(shellType)!;
            dynamic folder = shell.NameSpace("shell:AppsFolder");
            var apps = new List<InstalledApp>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var item in folder.Items())
            {
                string name = item.Name;
                string path = item.Path;
                if (name.Length == 0 || Noise.Any(n => name.Contains(n, StringComparison.OrdinalIgnoreCase)) || !seen.Add(name))
                {
                    continue;
                }
                apps.Add(new InstalledApp(name, path));
            }
            apps.Sort((a, b) => string.Compare(a.Name, b.Name, StringComparison.OrdinalIgnoreCase));
            _apps = apps;
            Directory.CreateDirectory(AppPaths.Root);
            File.WriteAllText(AppsCache, JsonSerializer.Serialize(apps));
            Log.Info($"{apps.Count} apps listed in {watch.ElapsedMilliseconds} ms");
        }
        catch (Exception error)
        {
            Log.Error("app list failed", error);
        }
    }
}
