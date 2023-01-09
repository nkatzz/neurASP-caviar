
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

	orientation(mws1gt_1,155,248). coords(mws1gt_1,109,121,248).  visible(mws1gt_1,248). happensAt(walking(mws1gt_1),248). orientation(mws1gt_3,157,248). coords(mws1gt_3,102,141,248).  visible(mws1gt_3,248). happensAt(walking(mws1gt_3),248).
orientation(mws1gt_1,155,249). coords(mws1gt_1,108,121,249).  visible(mws1gt_1,249). happensAt(walking(mws1gt_1),249). orientation(mws1gt_3,157,249). coords(mws1gt_3,102,142,249).  visible(mws1gt_3,249). happensAt(walking(mws1gt_3),249).
orientation(mws1gt_1,155,250). coords(mws1gt_1,106,120,250).  visible(mws1gt_1,250). happensAt(walking(mws1gt_1),250). orientation(mws1gt_3,157,250). coords(mws1gt_3,101,141,250).  visible(mws1gt_3,250). happensAt(walking(mws1gt_3),250).
orientation(mws1gt_1,155,251). coords(mws1gt_1,106,120,251).  visible(mws1gt_1,251). happensAt(walking(mws1gt_1),251). orientation(mws1gt_3,157,251). coords(mws1gt_3,100,141,251).  visible(mws1gt_3,251). happensAt(walking(mws1gt_3),251).
orientation(mws1gt_1,155,252). coords(mws1gt_1,106,120,252).  visible(mws1gt_1,252). happensAt(walking(mws1gt_1),252). orientation(mws1gt_3,157,252). coords(mws1gt_3,100,140,252).  visible(mws1gt_3,252). happensAt(walking(mws1gt_3),252).
orientation(mws1gt_1,155,253). coords(mws1gt_1,106,120,253).  visible(mws1gt_1,253). happensAt(walking(mws1gt_1),253). orientation(mws1gt_3,157,253). coords(mws1gt_3,99,140,253).  visible(mws1gt_3,253). happensAt(walking(mws1gt_3),253).
orientation(mws1gt_1,155,254). coords(mws1gt_1,106,119,254).  visible(mws1gt_1,254). happensAt(walking(mws1gt_1),254). orientation(mws1gt_3,157,254). coords(mws1gt_3,97,139,254).  visible(mws1gt_3,254). happensAt(walking(mws1gt_3),254).
orientation(mws1gt_1,155,255). coords(mws1gt_1,106,119,255).  visible(mws1gt_1,255). happensAt(walking(mws1gt_1),255). orientation(mws1gt_3,157,255). coords(mws1gt_3,96,139,255).  visible(mws1gt_3,255). happensAt(walking(mws1gt_3),255).
orientation(mws1gt_1,155,256). coords(mws1gt_1,106,119,256).  visible(mws1gt_1,256). happensAt(walking(mws1gt_1),256). orientation(mws1gt_3,157,256). coords(mws1gt_3,95,138,256).  visible(mws1gt_3,256). happensAt(walking(mws1gt_3),256).
orientation(mws1gt_1,155,257). coords(mws1gt_1,105,118,257).  visible(mws1gt_1,257). happensAt(walking(mws1gt_1),257). orientation(mws1gt_3,157,257). coords(mws1gt_3,94,139,257).  visible(mws1gt_3,257). happensAt(walking(mws1gt_3),257).
orientation(mws1gt_1,155,258). coords(mws1gt_1,104,117,258).  visible(mws1gt_1,258). happensAt(walking(mws1gt_1),258). orientation(mws1gt_3,157,258). coords(mws1gt_3,94,139,258).  visible(mws1gt_3,258). happensAt(walking(mws1gt_3),258).
orientation(mws1gt_1,155,259). coords(mws1gt_1,103,116,259).  visible(mws1gt_1,259). happensAt(walking(mws1gt_1),259). orientation(mws1gt_3,157,259). coords(mws1gt_3,93,139,259).  visible(mws1gt_3,259). happensAt(walking(mws1gt_3),259).
orientation(mws1gt_1,155,260). coords(mws1gt_1,102,115,260).  visible(mws1gt_1,260). happensAt(walking(mws1gt_1),260). orientation(mws1gt_3,157,260). coords(mws1gt_3,93,139,260).  visible(mws1gt_3,260). happensAt(walking(mws1gt_3),260).
orientation(mws1gt_1,155,261). coords(mws1gt_1,102,114,261).  visible(mws1gt_1,261). happensAt(walking(mws1gt_1),261). orientation(mws1gt_3,157,261). coords(mws1gt_3,93,138,261).  visible(mws1gt_3,261). happensAt(walking(mws1gt_3),261).
orientation(mws1gt_1,155,262). coords(mws1gt_1,101,113,262).  visible(mws1gt_1,262). happensAt(walking(mws1gt_1),262). orientation(mws1gt_3,157,262). coords(mws1gt_3,93,137,262).  visible(mws1gt_3,262). happensAt(walking(mws1gt_3),262).
orientation(mws1gt_1,155,263). coords(mws1gt_1,101,113,263).  visible(mws1gt_1,263). happensAt(walking(mws1gt_1),263). orientation(mws1gt_3,157,263). coords(mws1gt_3,92,137,263).  visible(mws1gt_3,263). happensAt(walking(mws1gt_3),263).
orientation(mws1gt_1,155,264). coords(mws1gt_1,100,112,264).  visible(mws1gt_1,264). happensAt(walking(mws1gt_1),264). orientation(mws1gt_3,157,264). coords(mws1gt_3,92,137,264).  visible(mws1gt_3,264). happensAt(walking(mws1gt_3),264).
orientation(mws1gt_1,155,265). coords(mws1gt_1,100,111,265).  visible(mws1gt_1,265). happensAt(walking(mws1gt_1),265). orientation(mws1gt_3,157,265). coords(mws1gt_3,92,136,265).  visible(mws1gt_3,265). happensAt(walking(mws1gt_3),265).
orientation(mws1gt_1,155,266). coords(mws1gt_1,100,111,266).  visible(mws1gt_1,266). happensAt(walking(mws1gt_1),266). orientation(mws1gt_3,157,266). coords(mws1gt_3,91,136,266).  visible(mws1gt_3,266). happensAt(walking(mws1gt_3),266).
orientation(mws1gt_1,155,267). coords(mws1gt_1,100,111,267).  visible(mws1gt_1,267). happensAt(walking(mws1gt_1),267). orientation(mws1gt_3,157,267). coords(mws1gt_3,90,135,267).  visible(mws1gt_3,267). happensAt(walking(mws1gt_3),267).
orientation(mws1gt_1,155,268). coords(mws1gt_1,100,111,268).  visible(mws1gt_1,268). happensAt(walking(mws1gt_1),268). orientation(mws1gt_3,157,268). coords(mws1gt_3,89,134,268).  visible(mws1gt_3,268). happensAt(walking(mws1gt_3),268).
orientation(mws1gt_1,155,269). coords(mws1gt_1,99,111,269).  visible(mws1gt_1,269). happensAt(walking(mws1gt_1),269). orientation(mws1gt_3,157,269). coords(mws1gt_3,88,134,269).  visible(mws1gt_3,269). happensAt(walking(mws1gt_3),269).
orientation(mws1gt_1,155,270). coords(mws1gt_1,99,111,270).  visible(mws1gt_1,270). happensAt(walking(mws1gt_1),270). orientation(mws1gt_3,157,270). coords(mws1gt_3,88,134,270).  visible(mws1gt_3,270). happensAt(walking(mws1gt_3),270).
orientation(mws1gt_1,155,271). coords(mws1gt_1,98,111,271).  visible(mws1gt_1,271). happensAt(walking(mws1gt_1),271). orientation(mws1gt_3,157,271). coords(mws1gt_3,88,133,271).  visible(mws1gt_3,271). happensAt(walking(mws1gt_3),271).
orientation(mws1gt_1,155,272). coords(mws1gt_1,97,110,272).  visible(mws1gt_1,272). happensAt(walking(mws1gt_1),272). orientation(mws1gt_3,157,272). coords(mws1gt_3,87,132,272).  visible(mws1gt_3,272). happensAt(walking(mws1gt_3),272).
#show holdsAt/2.