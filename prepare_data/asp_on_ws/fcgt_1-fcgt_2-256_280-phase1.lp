
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

	orientation(fcgt_1,157,256). coords(fcgt_1,152,150,256).  visible(fcgt_1,256). happensAt(walking(fcgt_1),256). orientation(fcgt_2,157,256). coords(fcgt_2,64,155,256).  visible(fcgt_2,256). happensAt(walking(fcgt_2),256).
orientation(fcgt_1,151,257). coords(fcgt_1,151,150,257).  visible(fcgt_1,257). happensAt(walking(fcgt_1),257). orientation(fcgt_2,157,257). coords(fcgt_2,64,154,257).  visible(fcgt_2,257). happensAt(walking(fcgt_2),257).
orientation(fcgt_1,151,258). coords(fcgt_1,149,149,258).  visible(fcgt_1,258). happensAt(walking(fcgt_1),258). orientation(fcgt_2,157,258). coords(fcgt_2,64,152,258).  visible(fcgt_2,258). happensAt(walking(fcgt_2),258).
orientation(fcgt_1,151,259). coords(fcgt_1,148,148,259).  visible(fcgt_1,259). happensAt(walking(fcgt_1),259). orientation(fcgt_2,157,259). coords(fcgt_2,65,151,259).  visible(fcgt_2,259). happensAt(walking(fcgt_2),259).
orientation(fcgt_1,151,260). coords(fcgt_1,145,146,260).  visible(fcgt_1,260). happensAt(walking(fcgt_1),260). orientation(fcgt_2,157,260). coords(fcgt_2,66,151,260).  visible(fcgt_2,260). happensAt(walking(fcgt_2),260).
orientation(fcgt_1,151,261). coords(fcgt_1,143,145,261).  visible(fcgt_1,261). happensAt(walking(fcgt_1),261). orientation(fcgt_2,157,261). coords(fcgt_2,67,150,261).  visible(fcgt_2,261). happensAt(walking(fcgt_2),261).
orientation(fcgt_1,151,262). coords(fcgt_1,140,143,262).  visible(fcgt_1,262). happensAt(walking(fcgt_1),262). orientation(fcgt_2,157,262). coords(fcgt_2,68,150,262).  visible(fcgt_2,262). happensAt(walking(fcgt_2),262).
orientation(fcgt_1,151,263). coords(fcgt_1,139,141,263).  visible(fcgt_1,263). happensAt(walking(fcgt_1),263). orientation(fcgt_2,157,263). coords(fcgt_2,68,149,263).  visible(fcgt_2,263). happensAt(walking(fcgt_2),263).
orientation(fcgt_1,151,264). coords(fcgt_1,138,139,264).  visible(fcgt_1,264). happensAt(walking(fcgt_1),264). orientation(fcgt_2,157,264). coords(fcgt_2,70,148,264).  visible(fcgt_2,264). happensAt(walking(fcgt_2),264).
orientation(fcgt_1,151,265). coords(fcgt_1,138,138,265).  visible(fcgt_1,265). happensAt(walking(fcgt_1),265). orientation(fcgt_2,157,265). coords(fcgt_2,70,147,265).  visible(fcgt_2,265). happensAt(walking(fcgt_2),265).
orientation(fcgt_1,151,266). coords(fcgt_1,137,138,266).  visible(fcgt_1,266). happensAt(walking(fcgt_1),266). orientation(fcgt_2,157,266). coords(fcgt_2,71,146,266).  visible(fcgt_2,266). happensAt(walking(fcgt_2),266).
orientation(fcgt_1,151,267). coords(fcgt_1,136,137,267).  visible(fcgt_1,267). happensAt(walking(fcgt_1),267). orientation(fcgt_2,157,267). coords(fcgt_2,71,146,267).  visible(fcgt_2,267). happensAt(walking(fcgt_2),267).
orientation(fcgt_1,151,268). coords(fcgt_1,135,137,268).  visible(fcgt_1,268). happensAt(walking(fcgt_1),268). orientation(fcgt_2,157,268). coords(fcgt_2,71,145,268).  visible(fcgt_2,268). happensAt(walking(fcgt_2),268).
orientation(fcgt_1,151,269). coords(fcgt_1,135,137,269).  visible(fcgt_1,269). happensAt(walking(fcgt_1),269). orientation(fcgt_2,157,269). coords(fcgt_2,71,145,269).  visible(fcgt_2,269). happensAt(walking(fcgt_2),269).
orientation(fcgt_1,151,270). coords(fcgt_1,133,137,270).  visible(fcgt_1,270). happensAt(walking(fcgt_1),270). orientation(fcgt_2,157,270). coords(fcgt_2,71,144,270).  visible(fcgt_2,270). happensAt(walking(fcgt_2),270).
orientation(fcgt_1,151,271). coords(fcgt_1,131,136,271).  visible(fcgt_1,271). happensAt(walking(fcgt_1),271). orientation(fcgt_2,157,271). coords(fcgt_2,73,144,271).  visible(fcgt_2,271). happensAt(walking(fcgt_2),271).
orientation(fcgt_1,151,272). coords(fcgt_1,130,137,272).  visible(fcgt_1,272). happensAt(walking(fcgt_1),272). orientation(fcgt_2,157,272). coords(fcgt_2,75,144,272).  visible(fcgt_2,272). happensAt(walking(fcgt_2),272).
orientation(fcgt_1,156,273). coords(fcgt_1,127,135,273).  visible(fcgt_1,273). happensAt(walking(fcgt_1),273). orientation(fcgt_2,157,273). coords(fcgt_2,75,143,273).  visible(fcgt_2,273). happensAt(walking(fcgt_2),273).
orientation(fcgt_1,156,274). coords(fcgt_1,125,133,274).  visible(fcgt_1,274). happensAt(walking(fcgt_1),274). orientation(fcgt_2,157,274). coords(fcgt_2,77,143,274).  visible(fcgt_2,274). happensAt(walking(fcgt_2),274).
orientation(fcgt_1,156,275). coords(fcgt_1,123,133,275).  visible(fcgt_1,275). happensAt(walking(fcgt_1),275). orientation(fcgt_2,157,275). coords(fcgt_2,77,143,275).  visible(fcgt_2,275). happensAt(walking(fcgt_2),275).
orientation(fcgt_1,153,276). coords(fcgt_1,122,133,276).  visible(fcgt_1,276). happensAt(walking(fcgt_1),276). orientation(fcgt_2,157,276). coords(fcgt_2,77,142,276).  visible(fcgt_2,276). happensAt(walking(fcgt_2),276).
orientation(fcgt_1,153,277). coords(fcgt_1,122,133,277).  visible(fcgt_1,277). happensAt(walking(fcgt_1),277). orientation(fcgt_2,157,277). coords(fcgt_2,77,141,277).  visible(fcgt_2,277). happensAt(walking(fcgt_2),277).
orientation(fcgt_1,153,278). coords(fcgt_1,121,133,278).  visible(fcgt_1,278). happensAt(walking(fcgt_1),278). orientation(fcgt_2,157,278). coords(fcgt_2,77,141,278).  visible(fcgt_2,278). happensAt(walking(fcgt_2),278).
orientation(fcgt_1,153,279). coords(fcgt_1,121,133,279).  visible(fcgt_1,279). happensAt(walking(fcgt_1),279). orientation(fcgt_2,157,279). coords(fcgt_2,78,140,279).  visible(fcgt_2,279). happensAt(walking(fcgt_2),279).
orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_2,150,280). coords(fcgt_2,78,138,280).  visible(fcgt_2,280). happensAt(walking(fcgt_2),280).
#show holdsAt/2.