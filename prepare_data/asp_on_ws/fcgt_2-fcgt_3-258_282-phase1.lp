
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

	orientation(fcgt_2,157,258). coords(fcgt_2,64,152,258).  visible(fcgt_2,258). happensAt(walking(fcgt_2),258). orientation(fcgt_3,166,258). coords(fcgt_3,100,178,258).  visible(fcgt_3,258). happensAt(walking(fcgt_3),258).
orientation(fcgt_2,157,259). coords(fcgt_2,65,151,259).  visible(fcgt_2,259). happensAt(walking(fcgt_2),259). orientation(fcgt_3,164,259). coords(fcgt_3,100,179,259).  visible(fcgt_3,259). happensAt(walking(fcgt_3),259).
orientation(fcgt_2,157,260). coords(fcgt_2,66,151,260).  visible(fcgt_2,260). happensAt(walking(fcgt_2),260). orientation(fcgt_3,165,260). coords(fcgt_3,100,179,260).  visible(fcgt_3,260). happensAt(walking(fcgt_3),260).
orientation(fcgt_2,157,261). coords(fcgt_2,67,150,261).  visible(fcgt_2,261). happensAt(walking(fcgt_2),261). orientation(fcgt_3,163,261). coords(fcgt_3,100,179,261).  visible(fcgt_3,261). happensAt(walking(fcgt_3),261).
orientation(fcgt_2,157,262). coords(fcgt_2,68,150,262).  visible(fcgt_2,262). happensAt(walking(fcgt_2),262). orientation(fcgt_3,166,262). coords(fcgt_3,100,179,262).  visible(fcgt_3,262). happensAt(walking(fcgt_3),262).
orientation(fcgt_2,157,263). coords(fcgt_2,68,149,263).  visible(fcgt_2,263). happensAt(walking(fcgt_2),263). orientation(fcgt_3,163,263). coords(fcgt_3,100,180,263).  visible(fcgt_3,263). happensAt(walking(fcgt_3),263).
orientation(fcgt_2,157,264). coords(fcgt_2,70,148,264).  visible(fcgt_2,264). happensAt(walking(fcgt_2),264). orientation(fcgt_3,165,264). coords(fcgt_3,100,180,264).  visible(fcgt_3,264). happensAt(walking(fcgt_3),264).
orientation(fcgt_2,157,265). coords(fcgt_2,70,147,265).  visible(fcgt_2,265). happensAt(walking(fcgt_2),265). orientation(fcgt_3,164,265). coords(fcgt_3,99,180,265).  visible(fcgt_3,265). happensAt(walking(fcgt_3),265).
orientation(fcgt_2,157,266). coords(fcgt_2,71,146,266).  visible(fcgt_2,266). happensAt(walking(fcgt_2),266). orientation(fcgt_3,163,266). coords(fcgt_3,99,181,266).  visible(fcgt_3,266). happensAt(walking(fcgt_3),266).
orientation(fcgt_2,157,267). coords(fcgt_2,71,146,267).  visible(fcgt_2,267). happensAt(walking(fcgt_2),267). orientation(fcgt_3,161,267). coords(fcgt_3,99,181,267).  visible(fcgt_3,267). happensAt(walking(fcgt_3),267).
orientation(fcgt_2,157,268). coords(fcgt_2,71,145,268).  visible(fcgt_2,268). happensAt(walking(fcgt_2),268). orientation(fcgt_3,163,268). coords(fcgt_3,99,181,268).  visible(fcgt_3,268). happensAt(walking(fcgt_3),268).
orientation(fcgt_2,157,269). coords(fcgt_2,71,145,269).  visible(fcgt_2,269). happensAt(walking(fcgt_2),269). orientation(fcgt_3,166,269). coords(fcgt_3,99,181,269).  visible(fcgt_3,269). happensAt(walking(fcgt_3),269).
orientation(fcgt_2,157,270). coords(fcgt_2,71,144,270).  visible(fcgt_2,270). happensAt(walking(fcgt_2),270). orientation(fcgt_3,164,270). coords(fcgt_3,98,181,270).  visible(fcgt_3,270). happensAt(walking(fcgt_3),270).
orientation(fcgt_2,157,271). coords(fcgt_2,73,144,271).  visible(fcgt_2,271). happensAt(walking(fcgt_2),271). orientation(fcgt_3,164,271). coords(fcgt_3,98,181,271).  visible(fcgt_3,271). happensAt(walking(fcgt_3),271).
orientation(fcgt_2,157,272). coords(fcgt_2,75,144,272).  visible(fcgt_2,272). happensAt(walking(fcgt_2),272). orientation(fcgt_3,164,272). coords(fcgt_3,98,181,272).  visible(fcgt_3,272). happensAt(walking(fcgt_3),272).
orientation(fcgt_2,157,273). coords(fcgt_2,75,143,273).  visible(fcgt_2,273). happensAt(walking(fcgt_2),273). orientation(fcgt_3,164,273). coords(fcgt_3,98,181,273).  visible(fcgt_3,273). happensAt(walking(fcgt_3),273).
orientation(fcgt_2,157,274). coords(fcgt_2,77,143,274).  visible(fcgt_2,274). happensAt(walking(fcgt_2),274). orientation(fcgt_3,164,274). coords(fcgt_3,98,181,274).  visible(fcgt_3,274). happensAt(walking(fcgt_3),274).
orientation(fcgt_2,157,275). coords(fcgt_2,77,143,275).  visible(fcgt_2,275). happensAt(walking(fcgt_2),275). orientation(fcgt_3,164,275). coords(fcgt_3,98,181,275).  visible(fcgt_3,275). happensAt(walking(fcgt_3),275).
orientation(fcgt_2,157,276). coords(fcgt_2,77,142,276).  visible(fcgt_2,276). happensAt(walking(fcgt_2),276). orientation(fcgt_3,164,276). coords(fcgt_3,98,182,276).  visible(fcgt_3,276). happensAt(walking(fcgt_3),276).
orientation(fcgt_2,157,277). coords(fcgt_2,77,141,277).  visible(fcgt_2,277). happensAt(walking(fcgt_2),277). orientation(fcgt_3,164,277). coords(fcgt_3,98,182,277).  visible(fcgt_3,277). happensAt(walking(fcgt_3),277).
orientation(fcgt_2,157,278). coords(fcgt_2,77,141,278).  visible(fcgt_2,278). happensAt(walking(fcgt_2),278). orientation(fcgt_3,164,278). coords(fcgt_3,98,182,278).  visible(fcgt_3,278). happensAt(walking(fcgt_3),278).
orientation(fcgt_2,157,279). coords(fcgt_2,78,140,279).  visible(fcgt_2,279). happensAt(walking(fcgt_2),279). orientation(fcgt_3,164,279). coords(fcgt_3,98,183,279).  visible(fcgt_3,279). happensAt(walking(fcgt_3),279).
orientation(fcgt_2,150,280). coords(fcgt_2,78,138,280).  visible(fcgt_2,280). happensAt(walking(fcgt_2),280). orientation(fcgt_3,164,280). coords(fcgt_3,98,183,280).  visible(fcgt_3,280). happensAt(walking(fcgt_3),280).
orientation(fcgt_2,150,281). coords(fcgt_2,78,138,281).  visible(fcgt_2,281). happensAt(walking(fcgt_2),281). orientation(fcgt_3,164,281). coords(fcgt_3,98,184,281).  visible(fcgt_3,281). happensAt(walking(fcgt_3),281).
orientation(fcgt_2,150,282). coords(fcgt_2,78,137,282).  visible(fcgt_2,282). happensAt(walking(fcgt_2),282). orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282).
#show holdsAt/2.