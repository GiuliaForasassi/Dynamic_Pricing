from gymnasium.envs.registration import register

register(
    id="DynamicPricing-v0",
    entry_point="pricing_advertising.environment.env:PricingEnv",
    order_enforce=False,
    disable_env_checker=True,
)
