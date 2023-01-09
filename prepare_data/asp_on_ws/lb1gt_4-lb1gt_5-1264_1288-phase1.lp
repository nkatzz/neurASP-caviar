
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_4,90,1264). coords(lb1gt_4,310,112,1264).  visible(lb1gt_4,1264). happensAt(inactive(lb1gt_4),1264). orientation(lb1gt_5,133,1264). coords(lb1gt_5,182,67,1264).  visible(lb1gt_5,1264). happensAt(walking(lb1gt_5),1264).
orientation(lb1gt_4,90,1265). coords(lb1gt_4,310,112,1265).  visible(lb1gt_4,1265). happensAt(inactive(lb1gt_4),1265). orientation(lb1gt_5,133,1265). coords(lb1gt_5,184,70,1265).  visible(lb1gt_5,1265). happensAt(walking(lb1gt_5),1265).
orientation(lb1gt_4,90,1266). coords(lb1gt_4,310,112,1266).  visible(lb1gt_4,1266). happensAt(inactive(lb1gt_4),1266). orientation(lb1gt_5,133,1266). coords(lb1gt_5,185,72,1266).  visible(lb1gt_5,1266). happensAt(walking(lb1gt_5),1266).
orientation(lb1gt_4,90,1267). coords(lb1gt_4,310,112,1267).  visible(lb1gt_4,1267). happensAt(inactive(lb1gt_4),1267). orientation(lb1gt_5,133,1267). coords(lb1gt_5,185,73,1267).  visible(lb1gt_5,1267). happensAt(walking(lb1gt_5),1267).
orientation(lb1gt_4,90,1268). coords(lb1gt_4,310,112,1268).  visible(lb1gt_4,1268). happensAt(inactive(lb1gt_4),1268). orientation(lb1gt_5,133,1268). coords(lb1gt_5,186,75,1268).  visible(lb1gt_5,1268). happensAt(walking(lb1gt_5),1268).
orientation(lb1gt_4,90,1269). coords(lb1gt_4,310,112,1269).  visible(lb1gt_4,1269). happensAt(inactive(lb1gt_4),1269). orientation(lb1gt_5,133,1269). coords(lb1gt_5,187,76,1269).  visible(lb1gt_5,1269). happensAt(walking(lb1gt_5),1269).
orientation(lb1gt_4,90,1270). coords(lb1gt_4,310,112,1270).  visible(lb1gt_4,1270). happensAt(inactive(lb1gt_4),1270). orientation(lb1gt_5,133,1270). coords(lb1gt_5,188,77,1270).  visible(lb1gt_5,1270). happensAt(walking(lb1gt_5),1270).
orientation(lb1gt_4,90,1271). coords(lb1gt_4,310,112,1271).  visible(lb1gt_4,1271). happensAt(inactive(lb1gt_4),1271). orientation(lb1gt_5,133,1271). coords(lb1gt_5,188,78,1271).  visible(lb1gt_5,1271). happensAt(walking(lb1gt_5),1271).
orientation(lb1gt_4,90,1272). coords(lb1gt_4,310,112,1272).  visible(lb1gt_4,1272). happensAt(inactive(lb1gt_4),1272). orientation(lb1gt_5,133,1272). coords(lb1gt_5,188,78,1272).  visible(lb1gt_5,1272). happensAt(walking(lb1gt_5),1272).
orientation(lb1gt_4,90,1273). coords(lb1gt_4,310,112,1273).  visible(lb1gt_4,1273). happensAt(inactive(lb1gt_4),1273). orientation(lb1gt_5,133,1273). coords(lb1gt_5,188,79,1273).  visible(lb1gt_5,1273). happensAt(walking(lb1gt_5),1273).
orientation(lb1gt_4,90,1274). coords(lb1gt_4,310,112,1274).  visible(lb1gt_4,1274). happensAt(inactive(lb1gt_4),1274). orientation(lb1gt_5,133,1274). coords(lb1gt_5,188,80,1274).  visible(lb1gt_5,1274). happensAt(walking(lb1gt_5),1274).
orientation(lb1gt_4,90,1275). coords(lb1gt_4,310,112,1275).  visible(lb1gt_4,1275). happensAt(inactive(lb1gt_4),1275). orientation(lb1gt_5,133,1275). coords(lb1gt_5,188,83,1275).  visible(lb1gt_5,1275). happensAt(walking(lb1gt_5),1275).
orientation(lb1gt_4,90,1276). coords(lb1gt_4,310,112,1276).  visible(lb1gt_4,1276). happensAt(inactive(lb1gt_4),1276). orientation(lb1gt_5,133,1276). coords(lb1gt_5,189,87,1276).  visible(lb1gt_5,1276). happensAt(walking(lb1gt_5),1276).
orientation(lb1gt_4,90,1277). coords(lb1gt_4,310,112,1277).  visible(lb1gt_4,1277). happensAt(inactive(lb1gt_4),1277). orientation(lb1gt_5,133,1277). coords(lb1gt_5,189,90,1277).  visible(lb1gt_5,1277). happensAt(walking(lb1gt_5),1277).
orientation(lb1gt_4,90,1278). coords(lb1gt_4,310,112,1278).  visible(lb1gt_4,1278). happensAt(inactive(lb1gt_4),1278). orientation(lb1gt_5,133,1278). coords(lb1gt_5,189,92,1278).  visible(lb1gt_5,1278). happensAt(walking(lb1gt_5),1278).
orientation(lb1gt_4,90,1279). coords(lb1gt_4,310,112,1279).  visible(lb1gt_4,1279). happensAt(inactive(lb1gt_4),1279). orientation(lb1gt_5,133,1279). coords(lb1gt_5,190,94,1279).  visible(lb1gt_5,1279). happensAt(walking(lb1gt_5),1279).
orientation(lb1gt_4,90,1280). coords(lb1gt_4,310,112,1280).  visible(lb1gt_4,1280). happensAt(inactive(lb1gt_4),1280). orientation(lb1gt_5,133,1280). coords(lb1gt_5,190,95,1280).  visible(lb1gt_5,1280). happensAt(walking(lb1gt_5),1280).
orientation(lb1gt_4,90,1281). coords(lb1gt_4,310,112,1281).  visible(lb1gt_4,1281). happensAt(inactive(lb1gt_4),1281). orientation(lb1gt_5,133,1281). coords(lb1gt_5,191,95,1281).  visible(lb1gt_5,1281). happensAt(walking(lb1gt_5),1281).
orientation(lb1gt_4,90,1282). coords(lb1gt_4,310,112,1282).  visible(lb1gt_4,1282). happensAt(inactive(lb1gt_4),1282). orientation(lb1gt_5,133,1282). coords(lb1gt_5,192,96,1282).  visible(lb1gt_5,1282). happensAt(walking(lb1gt_5),1282).
orientation(lb1gt_4,90,1283). coords(lb1gt_4,310,112,1283).  visible(lb1gt_4,1283). happensAt(inactive(lb1gt_4),1283). orientation(lb1gt_5,133,1283). coords(lb1gt_5,194,96,1283).  visible(lb1gt_5,1283). happensAt(walking(lb1gt_5),1283).
orientation(lb1gt_4,90,1284). coords(lb1gt_4,310,112,1284).  visible(lb1gt_4,1284). happensAt(inactive(lb1gt_4),1284). orientation(lb1gt_5,133,1284). coords(lb1gt_5,195,96,1284).  visible(lb1gt_5,1284). happensAt(walking(lb1gt_5),1284).
orientation(lb1gt_4,90,1285). coords(lb1gt_4,310,112,1285).  visible(lb1gt_4,1285). happensAt(inactive(lb1gt_4),1285). orientation(lb1gt_5,133,1285). coords(lb1gt_5,198,96,1285).  visible(lb1gt_5,1285). happensAt(walking(lb1gt_5),1285).
orientation(lb1gt_4,90,1286). coords(lb1gt_4,310,112,1286).  visible(lb1gt_4,1286). happensAt(inactive(lb1gt_4),1286). orientation(lb1gt_5,133,1286). coords(lb1gt_5,201,97,1286).  visible(lb1gt_5,1286). happensAt(walking(lb1gt_5),1286).
orientation(lb1gt_4,90,1287). coords(lb1gt_4,310,112,1287).  visible(lb1gt_4,1287). happensAt(inactive(lb1gt_4),1287). orientation(lb1gt_5,133,1287). coords(lb1gt_5,204,97,1287).  visible(lb1gt_5,1287). happensAt(walking(lb1gt_5),1287).
orientation(lb1gt_4,90,1288). coords(lb1gt_4,310,112,1288).  visible(lb1gt_4,1288). happensAt(inactive(lb1gt_4),1288). orientation(lb1gt_5,133,1288). coords(lb1gt_5,207,97,1288).  visible(lb1gt_5,1288). happensAt(walking(lb1gt_5),1288).
#show holdsAt/2.