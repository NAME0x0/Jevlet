using Jevlet.Core;

namespace Jevlet.Core.Tests;

public class CalculatorTests
{
    [Theory]
    [InlineData("18 percent of 240", "43.2")]
    [InlineData("15% of 80", "12")]
    [InlineData("12 times 37", "444")]
    [InlineData("2 to the power of 10", "1,024")]
    [InlineData("the square root of 144", "12")]
    [InlineData("250 divided by 7", "35.7143")]
    [InlineData("1024 / 16", "64")]
    [InlineData("what's 7 squared", "49")]
    [InlineData("(2 + 3) * 4", "20")]
    [InlineData("1,000 + 250", "1,250")]
    [InlineData("26 miles in km", "41.8429 km")]
    [InlineData("5 miles in km", "8.0467 km")]
    [InlineData("100 fahrenheit in celsius", "37.7778 °C")]
    [InlineData("3.5 kg in pounds", "7.7162 pounds")]
    [InlineData("convert 300 mph to km/h", "482.8032 km/h")]
    [InlineData("2 hours to minutes", "120 minutes")]
    public void Computes(string expression, string expected) => Assert.Equal(expected, Calculator.Evaluate(expression));

    [Theory]
    [InlineData("open spotify")]
    [InlineData("20 usd in aed")]
    [InlineData("5 miles in kg")]
    public void Declines_what_it_cannot_compute(string expression) => Assert.Null(Calculator.Evaluate(expression));

    [Fact]
    public void World_clock_knows_major_cities()
    {
        var utc = new DateTime(2026, 9, 25, 12, 0, 0, DateTimeKind.Utc);
        Assert.Equal(16, WorldClock.At("dubai", utc)!.Value.Local.Hour);
        Assert.Equal(21, WorldClock.At("Tokyo", utc)!.Value.Local.Hour);
        Assert.Equal("New York", WorldClock.At("new york", utc)!.Value.City);
        Assert.Null(WorldClock.At("atlantis", utc));
    }
}
