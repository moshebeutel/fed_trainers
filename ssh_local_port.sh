#!/bin/bash


if [ -z "$1" ]; then
  echo "No argument provided! Map to dsief08"
  dsi_num=8
else
  dsi_num="$1"
fi

if [ "$dsi_num" -lt 10 ]; then
  dsi_num="0${dsi_num}"
fi

port_num="60${dsi_num}"
dsi_name="dsief${dsi_num}"

echo "Map port ${port_num} to ${dsi_name}:22"

echo "DO NOT FORGET VPN!"

ssh -L "${port_num}:${dsi_name}:22" beutelm@dsihead.lnx.biu.ac.il