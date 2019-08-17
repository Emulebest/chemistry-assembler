defmodule DistributedAcoTest do
  use ExUnit.Case
  doctest DistributedAco

  test "Smoke test function 'hello'" do
    assert DistributedAco.hello == :world
  end
end
