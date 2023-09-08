
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

	orientation(fomdgt3_1,150,252). coords(fomdgt3_1,61,94,252).  visible(fomdgt3_1,252). happensAt(walking(fomdgt3_1),252). orientation(fomdgt3_2,130,252). coords(fomdgt3_2,228,128,252).  visible(fomdgt3_2,252). happensAt(walking(fomdgt3_2),252).
orientation(fomdgt3_1,150,253). coords(fomdgt3_1,61,94,253).  visible(fomdgt3_1,253). happensAt(walking(fomdgt3_1),253). orientation(fomdgt3_2,130,253). coords(fomdgt3_2,228,128,253).  visible(fomdgt3_2,253). happensAt(walking(fomdgt3_2),253).
orientation(fomdgt3_1,150,254). coords(fomdgt3_1,61,94,254).  visible(fomdgt3_1,254). happensAt(walking(fomdgt3_1),254). orientation(fomdgt3_2,130,254). coords(fomdgt3_2,240,139,254).  visible(fomdgt3_2,254). happensAt(walking(fomdgt3_2),254).
orientation(fomdgt3_1,150,255). coords(fomdgt3_1,61,94,255).  visible(fomdgt3_1,255). happensAt(walking(fomdgt3_1),255). orientation(fomdgt3_2,130,255). coords(fomdgt3_2,239,140,255).  visible(fomdgt3_2,255). happensAt(walking(fomdgt3_2),255).
orientation(fomdgt3_1,150,256). coords(fomdgt3_1,61,94,256).  visible(fomdgt3_1,256). happensAt(walking(fomdgt3_1),256). orientation(fomdgt3_2,130,256). coords(fomdgt3_2,239,140,256).  visible(fomdgt3_2,256). happensAt(walking(fomdgt3_2),256).
orientation(fomdgt3_1,150,257). coords(fomdgt3_1,61,94,257).  visible(fomdgt3_1,257). happensAt(walking(fomdgt3_1),257). orientation(fomdgt3_2,130,257). coords(fomdgt3_2,239,140,257).  visible(fomdgt3_2,257). happensAt(walking(fomdgt3_2),257).
orientation(fomdgt3_1,150,258). coords(fomdgt3_1,61,94,258).  visible(fomdgt3_1,258). happensAt(walking(fomdgt3_1),258). orientation(fomdgt3_2,130,258). coords(fomdgt3_2,239,140,258).  visible(fomdgt3_2,258). happensAt(walking(fomdgt3_2),258).
orientation(fomdgt3_1,150,259). coords(fomdgt3_1,61,94,259).  visible(fomdgt3_1,259). happensAt(walking(fomdgt3_1),259). orientation(fomdgt3_2,130,259). coords(fomdgt3_2,239,140,259).  visible(fomdgt3_2,259). happensAt(walking(fomdgt3_2),259).
orientation(fomdgt3_1,150,260). coords(fomdgt3_1,61,94,260).  visible(fomdgt3_1,260). happensAt(walking(fomdgt3_1),260). orientation(fomdgt3_2,130,260). coords(fomdgt3_2,239,140,260).  visible(fomdgt3_2,260). happensAt(walking(fomdgt3_2),260).
orientation(fomdgt3_1,150,261). coords(fomdgt3_1,59,96,261).  visible(fomdgt3_1,261). happensAt(walking(fomdgt3_1),261). orientation(fomdgt3_2,130,261). coords(fomdgt3_2,239,140,261).  visible(fomdgt3_2,261). happensAt(walking(fomdgt3_2),261).
orientation(fomdgt3_1,150,262). coords(fomdgt3_1,59,96,262).  visible(fomdgt3_1,262). happensAt(walking(fomdgt3_1),262). orientation(fomdgt3_2,130,262). coords(fomdgt3_2,239,140,262).  visible(fomdgt3_2,262). happensAt(walking(fomdgt3_2),262).
orientation(fomdgt3_1,150,263). coords(fomdgt3_1,59,96,263).  visible(fomdgt3_1,263). happensAt(walking(fomdgt3_1),263). orientation(fomdgt3_2,130,263). coords(fomdgt3_2,237,140,263).  visible(fomdgt3_2,263). happensAt(walking(fomdgt3_2),263).
orientation(fomdgt3_1,150,264). coords(fomdgt3_1,59,96,264).  visible(fomdgt3_1,264). happensAt(walking(fomdgt3_1),264). orientation(fomdgt3_2,130,264). coords(fomdgt3_2,237,140,264).  visible(fomdgt3_2,264). happensAt(walking(fomdgt3_2),264).
orientation(fomdgt3_1,150,265). coords(fomdgt3_1,59,96,265).  visible(fomdgt3_1,265). happensAt(walking(fomdgt3_1),265). orientation(fomdgt3_2,130,265). coords(fomdgt3_2,237,140,265).  visible(fomdgt3_2,265). happensAt(walking(fomdgt3_2),265).
orientation(fomdgt3_1,150,266). coords(fomdgt3_1,59,96,266).  visible(fomdgt3_1,266). happensAt(walking(fomdgt3_1),266). orientation(fomdgt3_2,130,266). coords(fomdgt3_2,237,140,266).  visible(fomdgt3_2,266). happensAt(walking(fomdgt3_2),266).
orientation(fomdgt3_1,150,267). coords(fomdgt3_1,58,96,267).  visible(fomdgt3_1,267). happensAt(walking(fomdgt3_1),267). orientation(fomdgt3_2,130,267). coords(fomdgt3_2,234,140,267).  visible(fomdgt3_2,267). happensAt(walking(fomdgt3_2),267).
orientation(fomdgt3_1,150,268). coords(fomdgt3_1,58,96,268).  visible(fomdgt3_1,268). happensAt(walking(fomdgt3_1),268). orientation(fomdgt3_2,130,268). coords(fomdgt3_2,234,140,268).  visible(fomdgt3_2,268). happensAt(walking(fomdgt3_2),268).
orientation(fomdgt3_1,150,269). coords(fomdgt3_1,57,96,269).  visible(fomdgt3_1,269). happensAt(walking(fomdgt3_1),269). orientation(fomdgt3_2,130,269). coords(fomdgt3_2,235,139,269).  visible(fomdgt3_2,269). happensAt(walking(fomdgt3_2),269).
orientation(fomdgt3_1,150,270). coords(fomdgt3_1,57,96,270).  visible(fomdgt3_1,270). happensAt(walking(fomdgt3_1),270). orientation(fomdgt3_2,130,270). coords(fomdgt3_2,235,135,270).  visible(fomdgt3_2,270). happensAt(walking(fomdgt3_2),270).
orientation(fomdgt3_1,150,271). coords(fomdgt3_1,57,96,271).  visible(fomdgt3_1,271). happensAt(walking(fomdgt3_1),271). orientation(fomdgt3_2,130,271). coords(fomdgt3_2,235,135,271).  visible(fomdgt3_2,271). happensAt(walking(fomdgt3_2),271).
orientation(fomdgt3_1,150,272). coords(fomdgt3_1,57,96,272).  visible(fomdgt3_1,272). happensAt(walking(fomdgt3_1),272). orientation(fomdgt3_2,130,272). coords(fomdgt3_2,235,135,272).  visible(fomdgt3_2,272). happensAt(walking(fomdgt3_2),272).
orientation(fomdgt3_1,150,273). coords(fomdgt3_1,53,96,273).  visible(fomdgt3_1,273). happensAt(walking(fomdgt3_1),273). orientation(fomdgt3_2,130,273). coords(fomdgt3_2,235,133,273).  visible(fomdgt3_2,273). happensAt(walking(fomdgt3_2),273).
orientation(fomdgt3_1,150,274). coords(fomdgt3_1,53,96,274).  visible(fomdgt3_1,274). happensAt(walking(fomdgt3_1),274). orientation(fomdgt3_2,130,274). coords(fomdgt3_2,236,133,274).  visible(fomdgt3_2,274). happensAt(walking(fomdgt3_2),274).
orientation(fomdgt3_1,150,275). coords(fomdgt3_1,53,96,275).  visible(fomdgt3_1,275). happensAt(walking(fomdgt3_1),275). orientation(fomdgt3_2,130,275). coords(fomdgt3_2,236,133,275).  visible(fomdgt3_2,275). happensAt(walking(fomdgt3_2),275).
orientation(fomdgt3_1,150,276). coords(fomdgt3_1,53,96,276).  visible(fomdgt3_1,276). happensAt(walking(fomdgt3_1),276). orientation(fomdgt3_2,130,276). coords(fomdgt3_2,235,132,276).  visible(fomdgt3_2,276). happensAt(walking(fomdgt3_2),276).
#show holdsAt/2.