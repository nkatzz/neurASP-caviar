
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

	orientation(fomdgt3_1,150,258). coords(fomdgt3_1,61,94,258).  visible(fomdgt3_1,258). happensAt(walking(fomdgt3_1),258). orientation(fomdgt3_6,130,258). coords(fomdgt3_6,206,150,258).  visible(fomdgt3_6,258). happensAt(walking(fomdgt3_6),258).
orientation(fomdgt3_1,150,259). coords(fomdgt3_1,61,94,259).  visible(fomdgt3_1,259). happensAt(walking(fomdgt3_1),259). orientation(fomdgt3_6,130,259). coords(fomdgt3_6,206,150,259).  visible(fomdgt3_6,259). happensAt(walking(fomdgt3_6),259).
orientation(fomdgt3_1,150,260). coords(fomdgt3_1,61,94,260).  visible(fomdgt3_1,260). happensAt(walking(fomdgt3_1),260). orientation(fomdgt3_6,130,260). coords(fomdgt3_6,206,156,260).  visible(fomdgt3_6,260). happensAt(walking(fomdgt3_6),260).
orientation(fomdgt3_1,150,261). coords(fomdgt3_1,59,96,261).  visible(fomdgt3_1,261). happensAt(walking(fomdgt3_1),261). orientation(fomdgt3_6,130,261). coords(fomdgt3_6,209,156,261).  visible(fomdgt3_6,261). happensAt(walking(fomdgt3_6),261).
orientation(fomdgt3_1,150,262). coords(fomdgt3_1,59,96,262).  visible(fomdgt3_1,262). happensAt(walking(fomdgt3_1),262). orientation(fomdgt3_6,130,262). coords(fomdgt3_6,209,156,262).  visible(fomdgt3_6,262). happensAt(walking(fomdgt3_6),262).
orientation(fomdgt3_1,150,263). coords(fomdgt3_1,59,96,263).  visible(fomdgt3_1,263). happensAt(walking(fomdgt3_1),263). orientation(fomdgt3_6,130,263). coords(fomdgt3_6,214,161,263).  visible(fomdgt3_6,263). happensAt(walking(fomdgt3_6),263).
orientation(fomdgt3_1,150,264). coords(fomdgt3_1,59,96,264).  visible(fomdgt3_1,264). happensAt(walking(fomdgt3_1),264). orientation(fomdgt3_6,130,264). coords(fomdgt3_6,218,162,264).  visible(fomdgt3_6,264). happensAt(walking(fomdgt3_6),264).
orientation(fomdgt3_1,150,265). coords(fomdgt3_1,59,96,265).  visible(fomdgt3_1,265). happensAt(walking(fomdgt3_1),265). orientation(fomdgt3_6,130,265). coords(fomdgt3_6,220,162,265).  visible(fomdgt3_6,265). happensAt(walking(fomdgt3_6),265).
orientation(fomdgt3_1,150,266). coords(fomdgt3_1,59,96,266).  visible(fomdgt3_1,266). happensAt(walking(fomdgt3_1),266). orientation(fomdgt3_6,130,266). coords(fomdgt3_6,220,162,266).  visible(fomdgt3_6,266). happensAt(walking(fomdgt3_6),266).
orientation(fomdgt3_1,150,267). coords(fomdgt3_1,58,96,267).  visible(fomdgt3_1,267). happensAt(walking(fomdgt3_1),267). orientation(fomdgt3_6,130,267). coords(fomdgt3_6,223,164,267).  visible(fomdgt3_6,267). happensAt(walking(fomdgt3_6),267).
orientation(fomdgt3_1,150,268). coords(fomdgt3_1,58,96,268).  visible(fomdgt3_1,268). happensAt(walking(fomdgt3_1),268). orientation(fomdgt3_6,130,268). coords(fomdgt3_6,223,164,268).  visible(fomdgt3_6,268). happensAt(walking(fomdgt3_6),268).
orientation(fomdgt3_1,150,269). coords(fomdgt3_1,57,96,269).  visible(fomdgt3_1,269). happensAt(walking(fomdgt3_1),269). orientation(fomdgt3_6,130,269). coords(fomdgt3_6,223,164,269).  visible(fomdgt3_6,269). happensAt(walking(fomdgt3_6),269).
orientation(fomdgt3_1,150,270). coords(fomdgt3_1,57,96,270).  visible(fomdgt3_1,270). happensAt(walking(fomdgt3_1),270). orientation(fomdgt3_6,130,270). coords(fomdgt3_6,223,164,270).  visible(fomdgt3_6,270). happensAt(walking(fomdgt3_6),270).
orientation(fomdgt3_1,150,271). coords(fomdgt3_1,57,96,271).  visible(fomdgt3_1,271). happensAt(walking(fomdgt3_1),271). orientation(fomdgt3_6,130,271). coords(fomdgt3_6,223,164,271).  visible(fomdgt3_6,271). happensAt(walking(fomdgt3_6),271).
orientation(fomdgt3_1,150,272). coords(fomdgt3_1,57,96,272).  visible(fomdgt3_1,272). happensAt(walking(fomdgt3_1),272). orientation(fomdgt3_6,130,272). coords(fomdgt3_6,223,164,272).  visible(fomdgt3_6,272). happensAt(walking(fomdgt3_6),272).
orientation(fomdgt3_1,150,273). coords(fomdgt3_1,53,96,273).  visible(fomdgt3_1,273). happensAt(walking(fomdgt3_1),273). orientation(fomdgt3_6,130,273). coords(fomdgt3_6,223,162,273).  visible(fomdgt3_6,273). happensAt(walking(fomdgt3_6),273).
orientation(fomdgt3_1,150,274). coords(fomdgt3_1,53,96,274).  visible(fomdgt3_1,274). happensAt(walking(fomdgt3_1),274). orientation(fomdgt3_6,130,274). coords(fomdgt3_6,223,161,274).  visible(fomdgt3_6,274). happensAt(walking(fomdgt3_6),274).
orientation(fomdgt3_1,150,275). coords(fomdgt3_1,53,96,275).  visible(fomdgt3_1,275). happensAt(walking(fomdgt3_1),275). orientation(fomdgt3_6,130,275). coords(fomdgt3_6,223,162,275).  visible(fomdgt3_6,275). happensAt(walking(fomdgt3_6),275).
orientation(fomdgt3_1,150,276). coords(fomdgt3_1,53,96,276).  visible(fomdgt3_1,276). happensAt(walking(fomdgt3_1),276). orientation(fomdgt3_6,130,276). coords(fomdgt3_6,223,163,276).  visible(fomdgt3_6,276). happensAt(walking(fomdgt3_6),276).
orientation(fomdgt3_1,150,277). coords(fomdgt3_1,51,96,277).  visible(fomdgt3_1,277). happensAt(walking(fomdgt3_1),277). orientation(fomdgt3_6,130,277). coords(fomdgt3_6,223,163,277).  visible(fomdgt3_6,277). happensAt(walking(fomdgt3_6),277).
orientation(fomdgt3_1,150,278). coords(fomdgt3_1,51,96,278).  visible(fomdgt3_1,278). happensAt(walking(fomdgt3_1),278). orientation(fomdgt3_6,130,278). coords(fomdgt3_6,223,159,278).  visible(fomdgt3_6,278). happensAt(walking(fomdgt3_6),278).
orientation(fomdgt3_1,150,279). coords(fomdgt3_1,51,96,279).  visible(fomdgt3_1,279). happensAt(walking(fomdgt3_1),279). orientation(fomdgt3_6,130,279). coords(fomdgt3_6,219,155,279).  visible(fomdgt3_6,279). happensAt(walking(fomdgt3_6),279).
orientation(fomdgt3_1,150,280). coords(fomdgt3_1,51,96,280).  visible(fomdgt3_1,280). happensAt(walking(fomdgt3_1),280). orientation(fomdgt3_6,130,280). coords(fomdgt3_6,219,155,280).  visible(fomdgt3_6,280). happensAt(walking(fomdgt3_6),280).
orientation(fomdgt3_1,150,281). coords(fomdgt3_1,51,96,281).  visible(fomdgt3_1,281). happensAt(walking(fomdgt3_1),281). orientation(fomdgt3_6,130,281). coords(fomdgt3_6,219,153,281).  visible(fomdgt3_6,281). happensAt(walking(fomdgt3_6),281).
orientation(fomdgt3_1,150,282). coords(fomdgt3_1,51,96,282).  visible(fomdgt3_1,282). happensAt(walking(fomdgt3_1),282). orientation(fomdgt3_6,130,282). coords(fomdgt3_6,219,152,282).  visible(fomdgt3_6,282). happensAt(walking(fomdgt3_6),282).
#show holdsAt/2.