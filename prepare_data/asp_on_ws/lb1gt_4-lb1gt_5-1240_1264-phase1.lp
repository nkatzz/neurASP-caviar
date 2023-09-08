
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

	orientation(lb1gt_4,90,1240). coords(lb1gt_4,310,112,1240).  visible(lb1gt_4,1240). happensAt(inactive(lb1gt_4),1240). orientation(lb1gt_5,127,1240). coords(lb1gt_5,174,42,1240).  visible(lb1gt_5,1240). happensAt(walking(lb1gt_5),1240).
orientation(lb1gt_4,90,1241). coords(lb1gt_4,310,112,1241).  visible(lb1gt_4,1241). happensAt(inactive(lb1gt_4),1241). orientation(lb1gt_5,127,1241). coords(lb1gt_5,174,43,1241).  visible(lb1gt_5,1241). happensAt(walking(lb1gt_5),1241).
orientation(lb1gt_4,90,1242). coords(lb1gt_4,310,112,1242).  visible(lb1gt_4,1242). happensAt(inactive(lb1gt_4),1242). orientation(lb1gt_5,127,1242). coords(lb1gt_5,175,44,1242).  visible(lb1gt_5,1242). happensAt(walking(lb1gt_5),1242).
orientation(lb1gt_4,90,1243). coords(lb1gt_4,310,112,1243).  visible(lb1gt_4,1243). happensAt(inactive(lb1gt_4),1243). orientation(lb1gt_5,127,1243). coords(lb1gt_5,175,45,1243).  visible(lb1gt_5,1243). happensAt(walking(lb1gt_5),1243).
orientation(lb1gt_4,90,1244). coords(lb1gt_4,310,112,1244).  visible(lb1gt_4,1244). happensAt(inactive(lb1gt_4),1244). orientation(lb1gt_5,127,1244). coords(lb1gt_5,175,46,1244).  visible(lb1gt_5,1244). happensAt(walking(lb1gt_5),1244).
orientation(lb1gt_4,90,1245). coords(lb1gt_4,310,112,1245).  visible(lb1gt_4,1245). happensAt(inactive(lb1gt_4),1245). orientation(lb1gt_5,127,1245). coords(lb1gt_5,175,46,1245).  visible(lb1gt_5,1245). happensAt(walking(lb1gt_5),1245).
orientation(lb1gt_4,90,1246). coords(lb1gt_4,310,112,1246).  visible(lb1gt_4,1246). happensAt(inactive(lb1gt_4),1246). orientation(lb1gt_5,127,1246). coords(lb1gt_5,175,47,1246).  visible(lb1gt_5,1246). happensAt(walking(lb1gt_5),1246).
orientation(lb1gt_4,90,1247). coords(lb1gt_4,310,112,1247).  visible(lb1gt_4,1247). happensAt(inactive(lb1gt_4),1247). orientation(lb1gt_5,127,1247). coords(lb1gt_5,175,47,1247).  visible(lb1gt_5,1247). happensAt(walking(lb1gt_5),1247).
orientation(lb1gt_4,90,1248). coords(lb1gt_4,310,112,1248).  visible(lb1gt_4,1248). happensAt(inactive(lb1gt_4),1248). orientation(lb1gt_5,127,1248). coords(lb1gt_5,175,47,1248).  visible(lb1gt_5,1248). happensAt(walking(lb1gt_5),1248).
orientation(lb1gt_4,90,1249). coords(lb1gt_4,310,112,1249).  visible(lb1gt_4,1249). happensAt(inactive(lb1gt_4),1249). orientation(lb1gt_5,127,1249). coords(lb1gt_5,175,49,1249).  visible(lb1gt_5,1249). happensAt(walking(lb1gt_5),1249).
orientation(lb1gt_4,90,1250). coords(lb1gt_4,310,112,1250).  visible(lb1gt_4,1250). happensAt(inactive(lb1gt_4),1250). orientation(lb1gt_5,127,1250). coords(lb1gt_5,175,52,1250).  visible(lb1gt_5,1250). happensAt(walking(lb1gt_5),1250).
orientation(lb1gt_4,90,1251). coords(lb1gt_4,310,112,1251).  visible(lb1gt_4,1251). happensAt(inactive(lb1gt_4),1251). orientation(lb1gt_5,127,1251). coords(lb1gt_5,175,53,1251).  visible(lb1gt_5,1251). happensAt(walking(lb1gt_5),1251).
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
#show holdsAt/2.