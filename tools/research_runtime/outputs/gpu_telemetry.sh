#!/bin/bash
# R9700 (PCI 03:00.0) sensors every 15 s, for evidence if the card drops off the bus again:
# time, edge/junction/mem C, PPT W, sclk/mclk MHz, fan rpm. Reads sysfs only.
D=/sys/bus/pci/devices/0000:03:00.0
H=$(ls -d $D/hwmon/hwmon* | head -1)
echo "# $(date -Is) crit edge/junction/mem C: $(($(cat $H/temp1_crit)/1000)) $(($(cat $H/temp2_crit)/1000)) $(($(cat $H/temp3_crit)/1000)), PPT cap W: $(($(cat $H/power1_cap)/1000000))"
while true; do
  r() { cat $H/$1 2>/dev/null || echo -1; }
  echo "$(date +%T) $(($(r temp1_input)/1000)) $(($(r temp2_input)/1000)) $(($(r temp3_input)/1000)) $(($(r power1_average)/1000000)) $(($(r freq1_input)/1000000)) $(($(r freq2_input)/1000000)) $(r fan1_input) $(cat $D/power/runtime_status)"
  sleep 15
done
