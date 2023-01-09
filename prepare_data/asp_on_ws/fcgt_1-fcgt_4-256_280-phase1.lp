
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

	orientation(fcgt_1,157,256). coords(fcgt_1,152,150,256).  visible(fcgt_1,256). happensAt(walking(fcgt_1),256). orientation(fcgt_4,166,256). coords(fcgt_4,80,205,256).  visible(fcgt_4,256). happensAt(inactive(fcgt_4),256).
orientation(fcgt_1,151,257). coords(fcgt_1,151,150,257).  visible(fcgt_1,257). happensAt(walking(fcgt_1),257). orientation(fcgt_4,166,257). coords(fcgt_4,80,205,257).  visible(fcgt_4,257). happensAt(inactive(fcgt_4),257).
orientation(fcgt_1,151,258). coords(fcgt_1,149,149,258).  visible(fcgt_1,258). happensAt(walking(fcgt_1),258). orientation(fcgt_4,166,258). coords(fcgt_4,80,205,258).  visible(fcgt_4,258). happensAt(inactive(fcgt_4),258).
orientation(fcgt_1,151,259). coords(fcgt_1,148,148,259).  visible(fcgt_1,259). happensAt(walking(fcgt_1),259). orientation(fcgt_4,166,259). coords(fcgt_4,80,205,259).  visible(fcgt_4,259). happensAt(inactive(fcgt_4),259).
orientation(fcgt_1,151,260). coords(fcgt_1,145,146,260).  visible(fcgt_1,260). happensAt(walking(fcgt_1),260). orientation(fcgt_4,166,260). coords(fcgt_4,80,205,260).  visible(fcgt_4,260). happensAt(inactive(fcgt_4),260).
orientation(fcgt_1,151,261). coords(fcgt_1,143,145,261).  visible(fcgt_1,261). happensAt(walking(fcgt_1),261). orientation(fcgt_4,166,261). coords(fcgt_4,80,205,261).  visible(fcgt_4,261). happensAt(inactive(fcgt_4),261).
orientation(fcgt_1,151,262). coords(fcgt_1,140,143,262).  visible(fcgt_1,262). happensAt(walking(fcgt_1),262). orientation(fcgt_4,166,262). coords(fcgt_4,80,205,262).  visible(fcgt_4,262). happensAt(inactive(fcgt_4),262).
orientation(fcgt_1,151,263). coords(fcgt_1,139,141,263).  visible(fcgt_1,263). happensAt(walking(fcgt_1),263). orientation(fcgt_4,166,263). coords(fcgt_4,81,205,263).  visible(fcgt_4,263). happensAt(inactive(fcgt_4),263).
orientation(fcgt_1,151,264). coords(fcgt_1,138,139,264).  visible(fcgt_1,264). happensAt(walking(fcgt_1),264). orientation(fcgt_4,166,264). coords(fcgt_4,81,205,264).  visible(fcgt_4,264). happensAt(inactive(fcgt_4),264).
orientation(fcgt_1,151,265). coords(fcgt_1,138,138,265).  visible(fcgt_1,265). happensAt(walking(fcgt_1),265). orientation(fcgt_4,166,265). coords(fcgt_4,81,205,265).  visible(fcgt_4,265). happensAt(inactive(fcgt_4),265).
orientation(fcgt_1,151,266). coords(fcgt_1,137,138,266).  visible(fcgt_1,266). happensAt(walking(fcgt_1),266). orientation(fcgt_4,166,266). coords(fcgt_4,81,205,266).  visible(fcgt_4,266). happensAt(inactive(fcgt_4),266).
orientation(fcgt_1,151,267). coords(fcgt_1,136,137,267).  visible(fcgt_1,267). happensAt(walking(fcgt_1),267). orientation(fcgt_4,166,267). coords(fcgt_4,81,205,267).  visible(fcgt_4,267). happensAt(inactive(fcgt_4),267).
orientation(fcgt_1,151,268). coords(fcgt_1,135,137,268).  visible(fcgt_1,268). happensAt(walking(fcgt_1),268). orientation(fcgt_4,166,268). coords(fcgt_4,81,205,268).  visible(fcgt_4,268). happensAt(inactive(fcgt_4),268).
orientation(fcgt_1,151,269). coords(fcgt_1,135,137,269).  visible(fcgt_1,269). happensAt(walking(fcgt_1),269). orientation(fcgt_4,166,269). coords(fcgt_4,81,205,269).  visible(fcgt_4,269). happensAt(inactive(fcgt_4),269).
orientation(fcgt_1,151,270). coords(fcgt_1,133,137,270).  visible(fcgt_1,270). happensAt(walking(fcgt_1),270). orientation(fcgt_4,166,270). coords(fcgt_4,81,205,270).  visible(fcgt_4,270). happensAt(inactive(fcgt_4),270).
orientation(fcgt_1,151,271). coords(fcgt_1,131,136,271).  visible(fcgt_1,271). happensAt(walking(fcgt_1),271). orientation(fcgt_4,166,271). coords(fcgt_4,81,205,271).  visible(fcgt_4,271). happensAt(inactive(fcgt_4),271).
orientation(fcgt_1,151,272). coords(fcgt_1,130,137,272).  visible(fcgt_1,272). happensAt(walking(fcgt_1),272). orientation(fcgt_4,166,272). coords(fcgt_4,81,205,272).  visible(fcgt_4,272). happensAt(inactive(fcgt_4),272).
orientation(fcgt_1,156,273). coords(fcgt_1,127,135,273).  visible(fcgt_1,273). happensAt(walking(fcgt_1),273). orientation(fcgt_4,166,273). coords(fcgt_4,81,205,273).  visible(fcgt_4,273). happensAt(inactive(fcgt_4),273).
orientation(fcgt_1,156,274). coords(fcgt_1,125,133,274).  visible(fcgt_1,274). happensAt(walking(fcgt_1),274). orientation(fcgt_4,166,274). coords(fcgt_4,81,205,274).  visible(fcgt_4,274). happensAt(inactive(fcgt_4),274).
orientation(fcgt_1,156,275). coords(fcgt_1,123,133,275).  visible(fcgt_1,275). happensAt(walking(fcgt_1),275). orientation(fcgt_4,166,275). coords(fcgt_4,81,205,275).  visible(fcgt_4,275). happensAt(inactive(fcgt_4),275).
orientation(fcgt_1,153,276). coords(fcgt_1,122,133,276).  visible(fcgt_1,276). happensAt(walking(fcgt_1),276). orientation(fcgt_4,166,276). coords(fcgt_4,81,205,276).  visible(fcgt_4,276). happensAt(inactive(fcgt_4),276).
orientation(fcgt_1,153,277). coords(fcgt_1,122,133,277).  visible(fcgt_1,277). happensAt(walking(fcgt_1),277). orientation(fcgt_4,166,277). coords(fcgt_4,81,205,277).  visible(fcgt_4,277). happensAt(inactive(fcgt_4),277).
orientation(fcgt_1,153,278). coords(fcgt_1,121,133,278).  visible(fcgt_1,278). happensAt(walking(fcgt_1),278). orientation(fcgt_4,166,278). coords(fcgt_4,81,205,278).  visible(fcgt_4,278). happensAt(inactive(fcgt_4),278).
orientation(fcgt_1,153,279). coords(fcgt_1,121,133,279).  visible(fcgt_1,279). happensAt(walking(fcgt_1),279). orientation(fcgt_4,166,279). coords(fcgt_4,81,205,279).  visible(fcgt_4,279). happensAt(inactive(fcgt_4),279).
orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_4,166,280). coords(fcgt_4,81,205,280).  visible(fcgt_4,280). happensAt(inactive(fcgt_4),280).
#show holdsAt/2.