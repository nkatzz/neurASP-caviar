
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

	orientation(fra1gt_6,123,258). coords(fra1gt_6,228,51,258).  visible(fra1gt_6,258). happensAt(walking(fra1gt_6),258). orientation(fra1gt_7,137,258). coords(fra1gt_7,148,41,258).  visible(fra1gt_7,258). happensAt(walking(fra1gt_7),258).
orientation(fra1gt_6,123,259). coords(fra1gt_6,227,52,259).  visible(fra1gt_6,259). happensAt(walking(fra1gt_6),259). orientation(fra1gt_7,137,259). coords(fra1gt_7,147,42,259).  visible(fra1gt_7,259). happensAt(walking(fra1gt_7),259).
orientation(fra1gt_6,123,260). coords(fra1gt_6,226,52,260).  visible(fra1gt_6,260). happensAt(walking(fra1gt_6),260). orientation(fra1gt_7,137,260). coords(fra1gt_7,147,42,260).  visible(fra1gt_7,260). happensAt(walking(fra1gt_7),260).
orientation(fra1gt_6,123,261). coords(fra1gt_6,224,52,261).  visible(fra1gt_6,261). happensAt(walking(fra1gt_6),261). orientation(fra1gt_7,137,261). coords(fra1gt_7,146,42,261).  visible(fra1gt_7,261). happensAt(walking(fra1gt_7),261).
orientation(fra1gt_6,123,262). coords(fra1gt_6,223,53,262).  visible(fra1gt_6,262). happensAt(walking(fra1gt_6),262). orientation(fra1gt_7,137,262). coords(fra1gt_7,146,42,262).  visible(fra1gt_7,262). happensAt(walking(fra1gt_7),262).
orientation(fra1gt_6,123,263). coords(fra1gt_6,222,53,263).  visible(fra1gt_6,263). happensAt(walking(fra1gt_6),263). orientation(fra1gt_7,137,263). coords(fra1gt_7,144,43,263).  visible(fra1gt_7,263). happensAt(walking(fra1gt_7),263).
orientation(fra1gt_6,123,264). coords(fra1gt_6,221,54,264).  visible(fra1gt_6,264). happensAt(walking(fra1gt_6),264). orientation(fra1gt_7,137,264). coords(fra1gt_7,144,43,264).  visible(fra1gt_7,264). happensAt(walking(fra1gt_7),264).
orientation(fra1gt_6,123,265). coords(fra1gt_6,221,54,265).  visible(fra1gt_6,265). happensAt(walking(fra1gt_6),265). orientation(fra1gt_7,137,265). coords(fra1gt_7,143,43,265).  visible(fra1gt_7,265). happensAt(walking(fra1gt_7),265).
orientation(fra1gt_6,123,266). coords(fra1gt_6,220,56,266).  visible(fra1gt_6,266). happensAt(walking(fra1gt_6),266). orientation(fra1gt_7,137,266). coords(fra1gt_7,143,43,266).  visible(fra1gt_7,266). happensAt(walking(fra1gt_7),266).
orientation(fra1gt_6,123,267). coords(fra1gt_6,220,57,267).  visible(fra1gt_6,267). happensAt(walking(fra1gt_6),267). orientation(fra1gt_7,137,267). coords(fra1gt_7,143,44,267).  visible(fra1gt_7,267). happensAt(walking(fra1gt_7),267).
orientation(fra1gt_6,123,268). coords(fra1gt_6,219,58,268).  visible(fra1gt_6,268). happensAt(walking(fra1gt_6),268). orientation(fra1gt_7,137,268). coords(fra1gt_7,143,45,268).  visible(fra1gt_7,268). happensAt(walking(fra1gt_7),268).
orientation(fra1gt_6,123,269). coords(fra1gt_6,219,59,269).  visible(fra1gt_6,269). happensAt(walking(fra1gt_6),269). orientation(fra1gt_7,137,269). coords(fra1gt_7,143,45,269).  visible(fra1gt_7,269). happensAt(walking(fra1gt_7),269).
orientation(fra1gt_6,123,270). coords(fra1gt_6,218,60,270).  visible(fra1gt_6,270). happensAt(walking(fra1gt_6),270). orientation(fra1gt_7,137,270). coords(fra1gt_7,142,45,270).  visible(fra1gt_7,270). happensAt(walking(fra1gt_7),270).
orientation(fra1gt_6,123,271). coords(fra1gt_6,217,61,271).  visible(fra1gt_6,271). happensAt(walking(fra1gt_6),271). orientation(fra1gt_7,137,271). coords(fra1gt_7,142,45,271).  visible(fra1gt_7,271). happensAt(walking(fra1gt_7),271).
orientation(fra1gt_6,123,272). coords(fra1gt_6,216,61,272).  visible(fra1gt_6,272). happensAt(walking(fra1gt_6),272). orientation(fra1gt_7,137,272). coords(fra1gt_7,142,45,272).  visible(fra1gt_7,272). happensAt(walking(fra1gt_7),272).
orientation(fra1gt_6,125,273). coords(fra1gt_6,215,61,273).  visible(fra1gt_6,273). happensAt(walking(fra1gt_6),273). orientation(fra1gt_7,137,273). coords(fra1gt_7,142,45,273).  visible(fra1gt_7,273). happensAt(walking(fra1gt_7),273).
orientation(fra1gt_6,128,274). coords(fra1gt_6,213,62,274).  visible(fra1gt_6,274). happensAt(walking(fra1gt_6),274). orientation(fra1gt_7,137,274). coords(fra1gt_7,142,45,274).  visible(fra1gt_7,274). happensAt(walking(fra1gt_7),274).
orientation(fra1gt_6,127,275). coords(fra1gt_6,211,62,275).  visible(fra1gt_6,275). happensAt(walking(fra1gt_6),275). orientation(fra1gt_7,137,275). coords(fra1gt_7,141,45,275).  visible(fra1gt_7,275). happensAt(walking(fra1gt_7),275).
orientation(fra1gt_6,128,276). coords(fra1gt_6,210,63,276).  visible(fra1gt_6,276). happensAt(walking(fra1gt_6),276). orientation(fra1gt_7,137,276). coords(fra1gt_7,141,46,276).  visible(fra1gt_7,276). happensAt(walking(fra1gt_7),276).
orientation(fra1gt_6,128,277). coords(fra1gt_6,208,63,277).  visible(fra1gt_6,277). happensAt(walking(fra1gt_6),277). orientation(fra1gt_7,137,277). coords(fra1gt_7,141,46,277).  visible(fra1gt_7,277). happensAt(walking(fra1gt_7),277).
orientation(fra1gt_6,128,278). coords(fra1gt_6,206,64,278).  visible(fra1gt_6,278). happensAt(walking(fra1gt_6),278). orientation(fra1gt_7,137,278). coords(fra1gt_7,141,46,278).  visible(fra1gt_7,278). happensAt(walking(fra1gt_7),278).
orientation(fra1gt_6,128,279). coords(fra1gt_6,204,66,279).  visible(fra1gt_6,279). happensAt(walking(fra1gt_6),279). orientation(fra1gt_7,137,279). coords(fra1gt_7,141,46,279).  visible(fra1gt_7,279). happensAt(walking(fra1gt_7),279).
orientation(fra1gt_6,128,280). coords(fra1gt_6,203,68,280).  visible(fra1gt_6,280). happensAt(walking(fra1gt_6),280). orientation(fra1gt_7,137,280). coords(fra1gt_7,140,46,280).  visible(fra1gt_7,280). happensAt(walking(fra1gt_7),280).
orientation(fra1gt_6,128,281). coords(fra1gt_6,203,69,281).  visible(fra1gt_6,281). happensAt(walking(fra1gt_6),281). orientation(fra1gt_7,137,281). coords(fra1gt_7,140,47,281).  visible(fra1gt_7,281). happensAt(walking(fra1gt_7),281).
orientation(fra1gt_6,128,282). coords(fra1gt_6,202,70,282).  visible(fra1gt_6,282). happensAt(walking(fra1gt_6),282). orientation(fra1gt_7,137,282). coords(fra1gt_7,140,47,282).  visible(fra1gt_7,282). happensAt(walking(fra1gt_7),282).
#show holdsAt/2.