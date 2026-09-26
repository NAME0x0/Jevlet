using Jevlet.Core;

namespace Jevlet.App;

/// <summary>The app's long-lived parts. The planner appears once the model has loaded.</summary>
internal sealed class Services(Settings settings, Stores stores, Notifier notifier, Scheduler scheduler)
{
    public Settings Settings { get; } = settings;
    public Stores Stores { get; } = stores;
    public Notifier Notifier { get; } = notifier;
    public Scheduler Scheduler { get; } = scheduler;
    public Executor Executor { get; } = new(stores, scheduler, settings);

    private volatile Planner? _planner;
    public Planner? Planner
    {
        get => _planner;
        set => _planner = value;
    }

    public DecisionEngine? Engine { get; set; }
    public string? ModelProblem { get; set; }
    public string ModelLabel { get; set; } = "";
}
