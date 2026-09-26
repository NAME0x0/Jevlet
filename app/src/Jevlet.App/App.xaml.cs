using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using Jevlet.Core;
using Microsoft.Win32;
using Forms = System.Windows.Forms;

namespace Jevlet.App;

/// <summary>
/// Starts once per user session and lives in the tray. A second launch just opens the palette
/// of the running instance.
/// </summary>
public partial class App : Application
{
    private const string InstanceName = @"Local\Jevlet.App";
    private const string SummonName = @"Local\Jevlet.Summon";
    private const string RunKey = @"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run";

    private Mutex? _instance;
    private EventWaitHandle? _summon;
    private Services? _services;
    private PaletteWindow? _palette;
    private HotkeyService? _hotkey;
    private Forms.NotifyIcon? _tray;

    protected override void OnStartup(StartupEventArgs e)
    {
        _instance = new Mutex(true, InstanceName, out var first);
        if (!first)
        {
            // Already running: ask that instance to open its palette, then leave.
            using var existing = EventWaitHandle.OpenExisting(SummonName);
            existing.Set();
            Shutdown();
            return;
        }
        base.OnStartup(e);
        DispatcherUnhandledException += (_, args) =>
        {
            Log.Error("unhandled UI exception", args.Exception);
            args.Handled = true; // an always-on assistant must not vanish over one bad command
        };
        AppDomain.CurrentDomain.UnhandledException += (_, args) => Log.Error("unhandled exception", args.ExceptionObject as Exception);
        TaskScheduler.UnobservedTaskException += (_, args) =>
        {
            Log.Error("unobserved task exception", args.Exception);
            args.SetObserved();
        };
        Log.Info($"starting Jevlet {typeof(App).Assembly.GetName().Version} (pid {Environment.ProcessId})");

        var settings = Settings.Load();
        Theme.Apply(settings.Theme);
        var stores = new Stores(Stores.DefaultPath);
        var notifier = new Notifier();
        _services = new Services(settings, stores, notifier, new Scheduler(stores, notifier));
        Desktop.LoadApps();
        _palette = new PaletteWindow(_services);
        _hotkey = new HotkeyService(settings.Hotkey, () => _palette.Toggle());
        CreateTray(settings);
        ListenForSummons();
        ApplyStartup(settings.StartWithWindows);
        SystemEvents.UserPreferenceChanged += (_, args) =>
        {
            if (args.Category == UserPreferenceCategory.General)
            {
                Dispatcher.BeginInvoke(() => Theme.Apply(settings.Theme));
            }
        };
        LoadModel(settings);
        if (!File.Exists(Path.Combine(AppPaths.Root, "welcomed")))
        {
            File.WriteAllText(Path.Combine(AppPaths.Root, "welcomed"), DateTime.Now.ToString("O"));
            notifier.Show("Jevlet is ready", $"Press {_hotkey.Active} anywhere and type what you want.", Notifier.Kind.Info, TimeSpan.FromSeconds(10));
        }
    }

    private void LoadModel(Settings settings)
    {
        var services = _services!;
        var folder = settings.ResolveModelDirectory();
        if (folder is null)
        {
            services.ModelProblem = $"No model found. Put an exported model in {Path.Combine(AppPaths.Root, "models", "current")}.";
            Log.Error(services.ModelProblem);
            return;
        }
        try
        {
            // Never answer with a model trained for other skills or questions.
            var mismatch = ModelCompatibility.Problem(ModelManifest.Load(Path.Combine(folder, "model.json")), Catalogue.Shared);
            if (mismatch is not null)
            {
                services.ModelProblem = mismatch;
                Log.Error($"refusing model {folder}: {mismatch}");
                return;
            }
        }
        catch (Exception error) when (error is IOException or System.Text.Json.JsonException or InvalidDataException or KeyNotFoundException)
        {
            services.ModelProblem = $"The model description in {folder} could not be read: {error.Message}";
            Log.Error("model.json unreadable", error);
            return;
        }
        Task.Run(() =>
        {
            try
            {
                var watch = Stopwatch.StartNew();
                var engine = DecisionEngine.Load(folder, settings.PreferGpu);
                engine.Recovered += reason =>
                {
                    Log.Info($"model recovered: {reason}");
                    services.ModelLabel = $"{Path.GetFileName(folder)} · {engine.Device.ToUpperInvariant()}";
                };
                var planner = new Planner(engine);
                planner.Plan("open settings", Snapshot.Empty, DateTime.Now); // warm-up: first run compiles kernels
                services.Engine = engine;
                services.ModelLabel = $"{Path.GetFileName(folder)} · {engine.Device.ToUpperInvariant()}";
                services.Planner = planner;
                Log.Info($"model {folder} ready on {engine.Device} in {watch.ElapsedMilliseconds} ms");
                watch.Restart();
                engine.WarmUp(); // usable meanwhile; this only removes first-use compile pauses
                Log.Info($"model warmed for common shapes in {watch.ElapsedMilliseconds} ms");
            }
            catch (Exception error)
            {
                services.ModelProblem = $"The model failed to load: {error.Message}";
                Log.Error("model load failed", error);
            }
        });
    }

    private void CreateTray(Settings settings)
    {
        var icon = GetResourceStream(new Uri("pack://application:,,,/Assets/jevlet.ico"))!.Stream;
        _tray = new Forms.NotifyIcon
        {
            Icon = new System.Drawing.Icon(icon),
            Text = $"Jevlet ({_hotkey!.Active})",
            Visible = true,
            ContextMenuStrip = new Forms.ContextMenuStrip(),
        };
        _tray.MouseClick += (_, args) =>
        {
            if (args.Button == Forms.MouseButtons.Left)
            {
                _palette!.Summon();
            }
        };
        var menu = _tray.ContextMenuStrip;
        menu.Items.Add($"Open Jevlet    {_hotkey.Active}", null, (_, _) => _palette!.Summon());
        menu.Items.Add(new Forms.ToolStripSeparator());
        var startup = new Forms.ToolStripMenuItem("Start with Windows") { Checked = settings.StartWithWindows, CheckOnClick = true };
        startup.CheckedChanged += (_, _) =>
        {
            settings.StartWithWindows = startup.Checked;
            settings.Save();
            ApplyStartup(startup.Checked);
        };
        menu.Items.Add(startup);
        menu.Items.Add("Settings file", null, (_, _) =>
        {
            settings.Save();
            Process.Start(new ProcessStartInfo(Path.Combine(AppPaths.Root, "settings.json")) { UseShellExecute = true });
        });
        menu.Items.Add("Log folder", null, (_, _) => Process.Start(new ProcessStartInfo(Log.Folder) { UseShellExecute = true }));
        menu.Items.Add(new Forms.ToolStripSeparator());
        menu.Items.Add("Quit", null, (_, _) => Quit());
    }

    private void ListenForSummons()
    {
        _summon = new EventWaitHandle(false, EventResetMode.AutoReset, SummonName);
        var thread = new Thread(() =>
        {
            while (_summon.WaitOne())
            {
                Dispatcher.BeginInvoke(() => _palette?.Summon());
            }
        })
        { IsBackground = true, Name = "Jevlet summon" };
        thread.Start();
    }

    private static void ApplyStartup(bool enabled)
    {
        try
        {
            if (enabled)
            {
                Registry.SetValue(RunKey, "Jevlet", $"\"{Environment.ProcessPath}\"");
            }
            else
            {
                using var key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Run", writable: true);
                key?.DeleteValue("Jevlet", throwOnMissingValue: false);
            }
        }
        catch (Exception error) when (error is UnauthorizedAccessException or IOException)
        {
            Log.Error("could not change start-with-Windows", error);
        }
    }

    private void Quit()
    {
        Log.Info("quitting");
        _hotkey?.Dispose();
        if (_tray is not null)
        {
            _tray.Visible = false;
            _tray.Dispose();
        }
        _services?.Engine?.Dispose();
        _services?.Stores.Dispose();
        Shutdown();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _instance?.Dispose();
        _summon?.Dispose();
        base.OnExit(e);
    }
}
