
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

	orientation(lb1gt_4,90,1252). coords(lb1gt_4,310,112,1252).  visible(lb1gt_4,1252). happensAt(inactive(lb1gt_4),1252). orientation(lb1gt_5,127,1252). coords(lb1gt_5,175,56,1252).  visible(lb1gt_5,1252). happensAt(walking(lb1gt_5),1252).
orientation(lb1gt_4,90,1253). coords(lb1gt_4,310,112,1253).  visible(lb1gt_4,1253). happensAt(inactive(lb1gt_4),1253). orientation(lb1gt_5,127,1253). coords(lb1gt_5,176,57,1253).  visible(lb1gt_5,1253). happensAt(walking(lb1gt_5),1253).
orientation(lb1gt_4,90,1254). coords(lb1gt_4,310,112,1254).  visible(lb1gt_4,1254). happensAt(inactive(lb1gt_4),1254). orientation(lb1gt_5,127,1254). coords(lb1gt_5,176,59,1254).  visible(lb1gt_5,1254). happensAt(walking(lb1gt_5),1254).
orientation(lb1gt_4,90,1255). coords(lb1gt_4,310,112,1255).  visible(lb1gt_4,1255). happensAt(inactive(lb1gt_4),1255). orientation(lb1gt_5,127,1255). coords(lb1gt_5,177,59,1255).  visible(lb1gt_5,1255). happensAt(walking(lb1gt_5),1255).
orientation(lb1gt_4,90,1256). coords(lb1gt_4,310,112,1256).  visible(lb1gt_4,1256). happensAt(inactive(lb1gt_4),1256). orientation(lb1gt_5,127,1256). coords(lb1gt_5,177,61,1256).  visible(lb1gt_5,1256). happensAt(walking(lb1gt_5),1256).
orientation(lb1gt_4,90,1257). coords(lb1gt_4,310,112,1257).  visible(lb1gt_4,1257). happensAt(inactive(lb1gt_4),1257). orientation(lb1gt_5,127,1257). coords(lb1gt_5,177,62,1257).  visible(lb1gt_5,1257). happensAt(walking(lb1gt_5),1257).
orientation(lb1gt_4,90,1258). coords(lb1gt_4,310,112,1258).  visible(lb1gt_4,1258). happensAt(inactive(lb1gt_4),1258). orientation(lb1gt_5,127,1258). coords(lb1gt_5,177,63,1258).  visible(lb1gt_5,1258). happensAt(walking(lb1gt_5),1258).
orientation(lb1gt_4,90,1259). coords(lb1gt_4,310,112,1259).  visible(lb1gt_4,1259). happensAt(inactive(lb1gt_4),1259). orientation(lb1gt_5,128,1259). coords(lb1gt_5,177,63,1259).  visible(lb1gt_5,1259). happensAt(walking(lb1gt_5),1259).
orientation(lb1gt_4,90,1260). coords(lb1gt_4,310,112,1260).  visible(lb1gt_4,1260). happensAt(inactive(lb1gt_4),1260). orientation(lb1gt_5,130,1260). coords(lb1gt_5,178,63,1260).  visible(lb1gt_5,1260). happensAt(walking(lb1gt_5),1260).
orientation(lb1gt_4,90,1261). coords(lb1gt_4,310,112,1261).  visible(lb1gt_4,1261). happensAt(inactive(lb1gt_4),1261). orientation(lb1gt_5,133,1261). coords(lb1gt_5,179,64,1261).  visible(lb1gt_5,1261). happensAt(walking(lb1gt_5),1261).
orientation(lb1gt_4,90,1262). coords(lb1gt_4,310,112,1262).  visible(lb1gt_4,1262). happensAt(inactive(lb1gt_4),1262). orientation(lb1gt_5,133,1262). coords(lb1gt_5,180,64,1262).  visible(lb1gt_5,1262). happensAt(walking(lb1gt_5),1262).
orientation(lb1gt_4,90,1263). coords(lb1gt_4,310,112,1263).  visible(lb1gt_4,1263). happensAt(inactive(lb1gt_4),1263). orientation(lb1gt_5,133,1263). coords(lb1gt_5,181,65,1263).  visible(lb1gt_5,1263). happensAt(walking(lb1gt_5),1263).
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
#show holdsAt/2.