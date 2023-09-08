
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

	orientation(fcgt_3,165,252). coords(fcgt_3,101,177,252).  visible(fcgt_3,252). happensAt(walking(fcgt_3),252). orientation(fcgt_4,166,252). coords(fcgt_4,80,205,252).  visible(fcgt_4,252). happensAt(inactive(fcgt_4),252).
orientation(fcgt_3,165,253). coords(fcgt_3,101,177,253).  visible(fcgt_3,253). happensAt(walking(fcgt_3),253). orientation(fcgt_4,166,253). coords(fcgt_4,80,205,253).  visible(fcgt_4,253). happensAt(inactive(fcgt_4),253).
orientation(fcgt_3,166,254). coords(fcgt_3,100,177,254).  visible(fcgt_3,254). happensAt(walking(fcgt_3),254). orientation(fcgt_4,166,254). coords(fcgt_4,80,205,254).  visible(fcgt_4,254). happensAt(inactive(fcgt_4),254).
orientation(fcgt_3,166,255). coords(fcgt_3,100,177,255).  visible(fcgt_3,255). happensAt(walking(fcgt_3),255). orientation(fcgt_4,166,255). coords(fcgt_4,80,205,255).  visible(fcgt_4,255). happensAt(inactive(fcgt_4),255).
orientation(fcgt_3,165,256). coords(fcgt_3,100,178,256).  visible(fcgt_3,256). happensAt(walking(fcgt_3),256). orientation(fcgt_4,166,256). coords(fcgt_4,80,205,256).  visible(fcgt_4,256). happensAt(inactive(fcgt_4),256).
orientation(fcgt_3,167,257). coords(fcgt_3,100,178,257).  visible(fcgt_3,257). happensAt(walking(fcgt_3),257). orientation(fcgt_4,166,257). coords(fcgt_4,80,205,257).  visible(fcgt_4,257). happensAt(inactive(fcgt_4),257).
orientation(fcgt_3,166,258). coords(fcgt_3,100,178,258).  visible(fcgt_3,258). happensAt(walking(fcgt_3),258). orientation(fcgt_4,166,258). coords(fcgt_4,80,205,258).  visible(fcgt_4,258). happensAt(inactive(fcgt_4),258).
orientation(fcgt_3,164,259). coords(fcgt_3,100,179,259).  visible(fcgt_3,259). happensAt(walking(fcgt_3),259). orientation(fcgt_4,166,259). coords(fcgt_4,80,205,259).  visible(fcgt_4,259). happensAt(inactive(fcgt_4),259).
orientation(fcgt_3,165,260). coords(fcgt_3,100,179,260).  visible(fcgt_3,260). happensAt(walking(fcgt_3),260). orientation(fcgt_4,166,260). coords(fcgt_4,80,205,260).  visible(fcgt_4,260). happensAt(inactive(fcgt_4),260).
orientation(fcgt_3,163,261). coords(fcgt_3,100,179,261).  visible(fcgt_3,261). happensAt(walking(fcgt_3),261). orientation(fcgt_4,166,261). coords(fcgt_4,80,205,261).  visible(fcgt_4,261). happensAt(inactive(fcgt_4),261).
orientation(fcgt_3,166,262). coords(fcgt_3,100,179,262).  visible(fcgt_3,262). happensAt(walking(fcgt_3),262). orientation(fcgt_4,166,262). coords(fcgt_4,80,205,262).  visible(fcgt_4,262). happensAt(inactive(fcgt_4),262).
orientation(fcgt_3,163,263). coords(fcgt_3,100,180,263).  visible(fcgt_3,263). happensAt(walking(fcgt_3),263). orientation(fcgt_4,166,263). coords(fcgt_4,81,205,263).  visible(fcgt_4,263). happensAt(inactive(fcgt_4),263).
orientation(fcgt_3,165,264). coords(fcgt_3,100,180,264).  visible(fcgt_3,264). happensAt(walking(fcgt_3),264). orientation(fcgt_4,166,264). coords(fcgt_4,81,205,264).  visible(fcgt_4,264). happensAt(inactive(fcgt_4),264).
orientation(fcgt_3,164,265). coords(fcgt_3,99,180,265).  visible(fcgt_3,265). happensAt(walking(fcgt_3),265). orientation(fcgt_4,166,265). coords(fcgt_4,81,205,265).  visible(fcgt_4,265). happensAt(inactive(fcgt_4),265).
orientation(fcgt_3,163,266). coords(fcgt_3,99,181,266).  visible(fcgt_3,266). happensAt(walking(fcgt_3),266). orientation(fcgt_4,166,266). coords(fcgt_4,81,205,266).  visible(fcgt_4,266). happensAt(inactive(fcgt_4),266).
orientation(fcgt_3,161,267). coords(fcgt_3,99,181,267).  visible(fcgt_3,267). happensAt(walking(fcgt_3),267). orientation(fcgt_4,166,267). coords(fcgt_4,81,205,267).  visible(fcgt_4,267). happensAt(inactive(fcgt_4),267).
orientation(fcgt_3,163,268). coords(fcgt_3,99,181,268).  visible(fcgt_3,268). happensAt(walking(fcgt_3),268). orientation(fcgt_4,166,268). coords(fcgt_4,81,205,268).  visible(fcgt_4,268). happensAt(inactive(fcgt_4),268).
orientation(fcgt_3,166,269). coords(fcgt_3,99,181,269).  visible(fcgt_3,269). happensAt(walking(fcgt_3),269). orientation(fcgt_4,166,269). coords(fcgt_4,81,205,269).  visible(fcgt_4,269). happensAt(inactive(fcgt_4),269).
orientation(fcgt_3,164,270). coords(fcgt_3,98,181,270).  visible(fcgt_3,270). happensAt(walking(fcgt_3),270). orientation(fcgt_4,166,270). coords(fcgt_4,81,205,270).  visible(fcgt_4,270). happensAt(inactive(fcgt_4),270).
orientation(fcgt_3,164,271). coords(fcgt_3,98,181,271).  visible(fcgt_3,271). happensAt(walking(fcgt_3),271). orientation(fcgt_4,166,271). coords(fcgt_4,81,205,271).  visible(fcgt_4,271). happensAt(inactive(fcgt_4),271).
orientation(fcgt_3,164,272). coords(fcgt_3,98,181,272).  visible(fcgt_3,272). happensAt(walking(fcgt_3),272). orientation(fcgt_4,166,272). coords(fcgt_4,81,205,272).  visible(fcgt_4,272). happensAt(inactive(fcgt_4),272).
orientation(fcgt_3,164,273). coords(fcgt_3,98,181,273).  visible(fcgt_3,273). happensAt(walking(fcgt_3),273). orientation(fcgt_4,166,273). coords(fcgt_4,81,205,273).  visible(fcgt_4,273). happensAt(inactive(fcgt_4),273).
orientation(fcgt_3,164,274). coords(fcgt_3,98,181,274).  visible(fcgt_3,274). happensAt(walking(fcgt_3),274). orientation(fcgt_4,166,274). coords(fcgt_4,81,205,274).  visible(fcgt_4,274). happensAt(inactive(fcgt_4),274).
orientation(fcgt_3,164,275). coords(fcgt_3,98,181,275).  visible(fcgt_3,275). happensAt(walking(fcgt_3),275). orientation(fcgt_4,166,275). coords(fcgt_4,81,205,275).  visible(fcgt_4,275). happensAt(inactive(fcgt_4),275).
orientation(fcgt_3,164,276). coords(fcgt_3,98,182,276).  visible(fcgt_3,276). happensAt(walking(fcgt_3),276). orientation(fcgt_4,166,276). coords(fcgt_4,81,205,276).  visible(fcgt_4,276). happensAt(inactive(fcgt_4),276).
#show holdsAt/2.