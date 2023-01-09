
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

	orientation(wk2gt_1,125,252). coords(wk2gt_1,115,53,252).  visible(wk2gt_1,252). happensAt(inactive(wk2gt_1),252). orientation(wk2gt_2,141,252). coords(wk2gt_2,98,62,252).  visible(wk2gt_2,252). happensAt(inactive(wk2gt_2),252).
orientation(wk2gt_1,125,253). coords(wk2gt_1,115,53,253).  visible(wk2gt_1,253). happensAt(inactive(wk2gt_1),253). orientation(wk2gt_2,141,253). coords(wk2gt_2,98,62,253).  visible(wk2gt_2,253). happensAt(inactive(wk2gt_2),253).
orientation(wk2gt_1,125,254). coords(wk2gt_1,115,53,254).  visible(wk2gt_1,254). happensAt(inactive(wk2gt_1),254). orientation(wk2gt_2,141,254). coords(wk2gt_2,98,62,254).  visible(wk2gt_2,254). happensAt(inactive(wk2gt_2),254).
orientation(wk2gt_1,125,255). coords(wk2gt_1,115,53,255).  visible(wk2gt_1,255). happensAt(inactive(wk2gt_1),255). orientation(wk2gt_2,141,255). coords(wk2gt_2,98,62,255).  visible(wk2gt_2,255). happensAt(inactive(wk2gt_2),255).
orientation(wk2gt_1,125,256). coords(wk2gt_1,115,53,256).  visible(wk2gt_1,256). happensAt(inactive(wk2gt_1),256). orientation(wk2gt_2,141,256). coords(wk2gt_2,98,62,256).  visible(wk2gt_2,256). happensAt(inactive(wk2gt_2),256).
orientation(wk2gt_1,125,257). coords(wk2gt_1,115,53,257).  visible(wk2gt_1,257). happensAt(inactive(wk2gt_1),257). orientation(wk2gt_2,141,257). coords(wk2gt_2,98,62,257).  visible(wk2gt_2,257). happensAt(inactive(wk2gt_2),257).
orientation(wk2gt_1,125,258). coords(wk2gt_1,115,53,258).  visible(wk2gt_1,258). happensAt(inactive(wk2gt_1),258). orientation(wk2gt_2,141,258). coords(wk2gt_2,98,62,258).  visible(wk2gt_2,258). happensAt(inactive(wk2gt_2),258).
orientation(wk2gt_1,125,259). coords(wk2gt_1,115,53,259).  visible(wk2gt_1,259). happensAt(inactive(wk2gt_1),259). orientation(wk2gt_2,141,259). coords(wk2gt_2,98,62,259).  visible(wk2gt_2,259). happensAt(inactive(wk2gt_2),259).
orientation(wk2gt_1,125,260). coords(wk2gt_1,115,53,260).  visible(wk2gt_1,260). happensAt(inactive(wk2gt_1),260). orientation(wk2gt_2,141,260). coords(wk2gt_2,98,62,260).  visible(wk2gt_2,260). happensAt(inactive(wk2gt_2),260).
orientation(wk2gt_1,125,261). coords(wk2gt_1,115,53,261).  visible(wk2gt_1,261). happensAt(inactive(wk2gt_1),261). orientation(wk2gt_2,141,261). coords(wk2gt_2,98,62,261).  visible(wk2gt_2,261). happensAt(inactive(wk2gt_2),261).
orientation(wk2gt_1,125,262). coords(wk2gt_1,115,53,262).  visible(wk2gt_1,262). happensAt(inactive(wk2gt_1),262). orientation(wk2gt_2,141,262). coords(wk2gt_2,98,62,262).  visible(wk2gt_2,262). happensAt(inactive(wk2gt_2),262).
orientation(wk2gt_1,125,263). coords(wk2gt_1,115,53,263).  visible(wk2gt_1,263). happensAt(inactive(wk2gt_1),263). orientation(wk2gt_2,141,263). coords(wk2gt_2,98,62,263).  visible(wk2gt_2,263). happensAt(inactive(wk2gt_2),263).
orientation(wk2gt_1,125,264). coords(wk2gt_1,115,53,264).  visible(wk2gt_1,264). happensAt(inactive(wk2gt_1),264). orientation(wk2gt_2,141,264). coords(wk2gt_2,98,62,264).  visible(wk2gt_2,264). happensAt(inactive(wk2gt_2),264).
orientation(wk2gt_1,125,265). coords(wk2gt_1,115,53,265).  visible(wk2gt_1,265). happensAt(inactive(wk2gt_1),265). orientation(wk2gt_2,141,265). coords(wk2gt_2,98,62,265).  visible(wk2gt_2,265). happensAt(inactive(wk2gt_2),265).
orientation(wk2gt_1,125,266). coords(wk2gt_1,115,53,266).  visible(wk2gt_1,266). happensAt(inactive(wk2gt_1),266). orientation(wk2gt_2,141,266). coords(wk2gt_2,98,62,266).  visible(wk2gt_2,266). happensAt(inactive(wk2gt_2),266).
orientation(wk2gt_1,125,267). coords(wk2gt_1,115,53,267).  visible(wk2gt_1,267). happensAt(inactive(wk2gt_1),267). orientation(wk2gt_2,141,267). coords(wk2gt_2,98,62,267).  visible(wk2gt_2,267). happensAt(inactive(wk2gt_2),267).
orientation(wk2gt_1,125,268). coords(wk2gt_1,115,53,268).  visible(wk2gt_1,268). happensAt(inactive(wk2gt_1),268). orientation(wk2gt_2,141,268). coords(wk2gt_2,98,62,268).  visible(wk2gt_2,268). happensAt(inactive(wk2gt_2),268).
orientation(wk2gt_1,125,269). coords(wk2gt_1,115,53,269).  visible(wk2gt_1,269). happensAt(inactive(wk2gt_1),269). orientation(wk2gt_2,141,269). coords(wk2gt_2,98,62,269).  visible(wk2gt_2,269). happensAt(inactive(wk2gt_2),269).
orientation(wk2gt_1,125,270). coords(wk2gt_1,115,53,270).  visible(wk2gt_1,270). happensAt(inactive(wk2gt_1),270). orientation(wk2gt_2,141,270). coords(wk2gt_2,98,62,270).  visible(wk2gt_2,270). happensAt(inactive(wk2gt_2),270).
orientation(wk2gt_1,125,271). coords(wk2gt_1,115,53,271).  visible(wk2gt_1,271). happensAt(inactive(wk2gt_1),271). orientation(wk2gt_2,141,271). coords(wk2gt_2,98,62,271).  visible(wk2gt_2,271). happensAt(inactive(wk2gt_2),271).
orientation(wk2gt_1,125,272). coords(wk2gt_1,115,53,272).  visible(wk2gt_1,272). happensAt(inactive(wk2gt_1),272). orientation(wk2gt_2,141,272). coords(wk2gt_2,98,62,272).  visible(wk2gt_2,272). happensAt(inactive(wk2gt_2),272).
orientation(wk2gt_1,125,273). coords(wk2gt_1,115,53,273).  visible(wk2gt_1,273). happensAt(inactive(wk2gt_1),273). orientation(wk2gt_2,141,273). coords(wk2gt_2,98,62,273).  visible(wk2gt_2,273). happensAt(inactive(wk2gt_2),273).
orientation(wk2gt_1,125,274). coords(wk2gt_1,115,53,274).  visible(wk2gt_1,274). happensAt(inactive(wk2gt_1),274). orientation(wk2gt_2,141,274). coords(wk2gt_2,98,62,274).  visible(wk2gt_2,274). happensAt(inactive(wk2gt_2),274).
orientation(wk2gt_1,125,275). coords(wk2gt_1,115,53,275).  visible(wk2gt_1,275). happensAt(inactive(wk2gt_1),275). orientation(wk2gt_2,141,275). coords(wk2gt_2,98,62,275).  visible(wk2gt_2,275). happensAt(inactive(wk2gt_2),275).
orientation(wk2gt_1,125,276). coords(wk2gt_1,115,53,276).  visible(wk2gt_1,276). happensAt(inactive(wk2gt_1),276). orientation(wk2gt_2,141,276). coords(wk2gt_2,98,63,276).  visible(wk2gt_2,276). happensAt(inactive(wk2gt_2),276).
#show holdsAt/2.