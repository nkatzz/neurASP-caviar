
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

	orientation(mws1gt_3,157,252). coords(mws1gt_3,100,140,252).  visible(mws1gt_3,252). happensAt(walking(mws1gt_3),252). orientation(mws1gt_6,146,252). coords(mws1gt_6,42,117,252).  visible(mws1gt_6,252). happensAt(walking(mws1gt_6),252).
orientation(mws1gt_3,157,253). coords(mws1gt_3,99,140,253).  visible(mws1gt_3,253). happensAt(walking(mws1gt_3),253). orientation(mws1gt_6,146,253). coords(mws1gt_6,42,119,253).  visible(mws1gt_6,253). happensAt(walking(mws1gt_6),253).
orientation(mws1gt_3,157,254). coords(mws1gt_3,97,139,254).  visible(mws1gt_3,254). happensAt(walking(mws1gt_3),254). orientation(mws1gt_6,146,254). coords(mws1gt_6,42,119,254).  visible(mws1gt_6,254). happensAt(walking(mws1gt_6),254).
orientation(mws1gt_3,157,255). coords(mws1gt_3,96,139,255).  visible(mws1gt_3,255). happensAt(walking(mws1gt_3),255). orientation(mws1gt_6,146,255). coords(mws1gt_6,43,120,255).  visible(mws1gt_6,255). happensAt(walking(mws1gt_6),255).
orientation(mws1gt_3,157,256). coords(mws1gt_3,95,138,256).  visible(mws1gt_3,256). happensAt(walking(mws1gt_3),256). orientation(mws1gt_6,146,256). coords(mws1gt_6,43,120,256).  visible(mws1gt_6,256). happensAt(walking(mws1gt_6),256).
orientation(mws1gt_3,157,257). coords(mws1gt_3,94,139,257).  visible(mws1gt_3,257). happensAt(walking(mws1gt_3),257). orientation(mws1gt_6,146,257). coords(mws1gt_6,43,121,257).  visible(mws1gt_6,257). happensAt(walking(mws1gt_6),257).
orientation(mws1gt_3,157,258). coords(mws1gt_3,94,139,258).  visible(mws1gt_3,258). happensAt(walking(mws1gt_3),258). orientation(mws1gt_6,146,258). coords(mws1gt_6,43,121,258).  visible(mws1gt_6,258). happensAt(walking(mws1gt_6),258).
orientation(mws1gt_3,157,259). coords(mws1gt_3,93,139,259).  visible(mws1gt_3,259). happensAt(walking(mws1gt_3),259). orientation(mws1gt_6,146,259). coords(mws1gt_6,43,124,259).  visible(mws1gt_6,259). happensAt(walking(mws1gt_6),259).
orientation(mws1gt_3,157,260). coords(mws1gt_3,93,139,260).  visible(mws1gt_3,260). happensAt(walking(mws1gt_3),260). orientation(mws1gt_6,146,260). coords(mws1gt_6,43,125,260).  visible(mws1gt_6,260). happensAt(walking(mws1gt_6),260).
orientation(mws1gt_3,157,261). coords(mws1gt_3,93,138,261).  visible(mws1gt_3,261). happensAt(walking(mws1gt_3),261). orientation(mws1gt_6,146,261). coords(mws1gt_6,43,126,261).  visible(mws1gt_6,261). happensAt(walking(mws1gt_6),261).
orientation(mws1gt_3,157,262). coords(mws1gt_3,93,137,262).  visible(mws1gt_3,262). happensAt(walking(mws1gt_3),262). orientation(mws1gt_6,146,262). coords(mws1gt_6,43,128,262).  visible(mws1gt_6,262). happensAt(walking(mws1gt_6),262).
orientation(mws1gt_3,157,263). coords(mws1gt_3,92,137,263).  visible(mws1gt_3,263). happensAt(walking(mws1gt_3),263). orientation(mws1gt_6,146,263). coords(mws1gt_6,42,129,263).  visible(mws1gt_6,263). happensAt(walking(mws1gt_6),263).
orientation(mws1gt_3,157,264). coords(mws1gt_3,92,137,264).  visible(mws1gt_3,264). happensAt(walking(mws1gt_3),264). orientation(mws1gt_6,146,264). coords(mws1gt_6,42,129,264).  visible(mws1gt_6,264). happensAt(walking(mws1gt_6),264).
orientation(mws1gt_3,157,265). coords(mws1gt_3,92,136,265).  visible(mws1gt_3,265). happensAt(walking(mws1gt_3),265). orientation(mws1gt_6,146,265). coords(mws1gt_6,42,129,265).  visible(mws1gt_6,265). happensAt(walking(mws1gt_6),265).
orientation(mws1gt_3,157,266). coords(mws1gt_3,91,136,266).  visible(mws1gt_3,266). happensAt(walking(mws1gt_3),266). orientation(mws1gt_6,146,266). coords(mws1gt_6,42,130,266).  visible(mws1gt_6,266). happensAt(walking(mws1gt_6),266).
orientation(mws1gt_3,157,267). coords(mws1gt_3,90,135,267).  visible(mws1gt_3,267). happensAt(walking(mws1gt_3),267). orientation(mws1gt_6,146,267). coords(mws1gt_6,42,130,267).  visible(mws1gt_6,267). happensAt(walking(mws1gt_6),267).
orientation(mws1gt_3,157,268). coords(mws1gt_3,89,134,268).  visible(mws1gt_3,268). happensAt(walking(mws1gt_3),268). orientation(mws1gt_6,146,268). coords(mws1gt_6,42,130,268).  visible(mws1gt_6,268). happensAt(walking(mws1gt_6),268).
orientation(mws1gt_3,157,269). coords(mws1gt_3,88,134,269).  visible(mws1gt_3,269). happensAt(walking(mws1gt_3),269). orientation(mws1gt_6,146,269). coords(mws1gt_6,42,131,269).  visible(mws1gt_6,269). happensAt(walking(mws1gt_6),269).
orientation(mws1gt_3,157,270). coords(mws1gt_3,88,134,270).  visible(mws1gt_3,270). happensAt(walking(mws1gt_3),270). orientation(mws1gt_6,146,270). coords(mws1gt_6,42,132,270).  visible(mws1gt_6,270). happensAt(walking(mws1gt_6),270).
orientation(mws1gt_3,157,271). coords(mws1gt_3,88,133,271).  visible(mws1gt_3,271). happensAt(walking(mws1gt_3),271). orientation(mws1gt_6,146,271). coords(mws1gt_6,41,133,271).  visible(mws1gt_6,271). happensAt(walking(mws1gt_6),271).
orientation(mws1gt_3,157,272). coords(mws1gt_3,87,132,272).  visible(mws1gt_3,272). happensAt(walking(mws1gt_3),272). orientation(mws1gt_6,146,272). coords(mws1gt_6,41,133,272).  visible(mws1gt_6,272). happensAt(walking(mws1gt_6),272).
orientation(mws1gt_3,157,273). coords(mws1gt_3,87,132,273).  visible(mws1gt_3,273). happensAt(walking(mws1gt_3),273). orientation(mws1gt_6,146,273). coords(mws1gt_6,40,134,273).  visible(mws1gt_6,273). happensAt(walking(mws1gt_6),273).
orientation(mws1gt_3,155,274). coords(mws1gt_3,87,132,274).  visible(mws1gt_3,274). happensAt(walking(mws1gt_3),274). orientation(mws1gt_6,146,274). coords(mws1gt_6,40,134,274).  visible(mws1gt_6,274). happensAt(walking(mws1gt_6),274).
orientation(mws1gt_3,155,275). coords(mws1gt_3,87,132,275).  visible(mws1gt_3,275). happensAt(walking(mws1gt_3),275). orientation(mws1gt_6,146,275). coords(mws1gt_6,40,135,275).  visible(mws1gt_6,275). happensAt(walking(mws1gt_6),275).
orientation(mws1gt_3,155,276). coords(mws1gt_3,87,132,276).  visible(mws1gt_3,276). happensAt(walking(mws1gt_3),276). orientation(mws1gt_6,146,276). coords(mws1gt_6,40,136,276).  visible(mws1gt_6,276). happensAt(walking(mws1gt_6),276).
#show holdsAt/2.