
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

	orientation(fra2gt_3,151,250). coords(fra2gt_3,62,144,250).  visible(fra2gt_3,250). happensAt(walking(fra2gt_3),250). orientation(fra2gt_4,155,250). coords(fra2gt_4,163,160,250).  visible(fra2gt_4,250). happensAt(walking(fra2gt_4),250).
orientation(fra2gt_3,151,251). coords(fra2gt_3,61,142,251).  visible(fra2gt_3,251). happensAt(walking(fra2gt_3),251). orientation(fra2gt_4,155,251). coords(fra2gt_4,162,158,251).  visible(fra2gt_4,251). happensAt(walking(fra2gt_4),251).
orientation(fra2gt_3,151,252). coords(fra2gt_3,61,142,252).  visible(fra2gt_3,252). happensAt(walking(fra2gt_3),252). orientation(fra2gt_4,155,252). coords(fra2gt_4,162,157,252).  visible(fra2gt_4,252). happensAt(walking(fra2gt_4),252).
orientation(fra2gt_3,151,253). coords(fra2gt_3,63,142,253).  visible(fra2gt_3,253). happensAt(walking(fra2gt_3),253). orientation(fra2gt_4,155,253). coords(fra2gt_4,161,155,253).  visible(fra2gt_4,253). happensAt(walking(fra2gt_4),253).
orientation(fra2gt_3,151,254). coords(fra2gt_3,64,141,254).  visible(fra2gt_3,254). happensAt(walking(fra2gt_3),254). orientation(fra2gt_4,155,254). coords(fra2gt_4,160,154,254).  visible(fra2gt_4,254). happensAt(walking(fra2gt_4),254).
orientation(fra2gt_3,151,255). coords(fra2gt_3,65,141,255).  visible(fra2gt_3,255). happensAt(walking(fra2gt_3),255). orientation(fra2gt_4,155,255). coords(fra2gt_4,160,153,255).  visible(fra2gt_4,255). happensAt(walking(fra2gt_4),255).
orientation(fra2gt_3,151,256). coords(fra2gt_3,66,141,256).  visible(fra2gt_3,256). happensAt(walking(fra2gt_3),256). orientation(fra2gt_4,155,256). coords(fra2gt_4,159,152,256).  visible(fra2gt_4,256). happensAt(walking(fra2gt_4),256).
orientation(fra2gt_3,151,257). coords(fra2gt_3,66,141,257).  visible(fra2gt_3,257). happensAt(walking(fra2gt_3),257). orientation(fra2gt_4,155,257). coords(fra2gt_4,158,151,257).  visible(fra2gt_4,257). happensAt(walking(fra2gt_4),257).
orientation(fra2gt_3,151,258). coords(fra2gt_3,67,140,258).  visible(fra2gt_3,258). happensAt(walking(fra2gt_3),258). orientation(fra2gt_4,155,258). coords(fra2gt_4,156,149,258).  visible(fra2gt_4,258). happensAt(walking(fra2gt_4),258).
orientation(fra2gt_3,151,259). coords(fra2gt_3,67,139,259).  visible(fra2gt_3,259). happensAt(walking(fra2gt_3),259). orientation(fra2gt_4,155,259). coords(fra2gt_4,153,148,259).  visible(fra2gt_4,259). happensAt(walking(fra2gt_4),259).
orientation(fra2gt_3,151,260). coords(fra2gt_3,67,138,260).  visible(fra2gt_3,260). happensAt(walking(fra2gt_3),260). orientation(fra2gt_4,155,260). coords(fra2gt_4,151,147,260).  visible(fra2gt_4,260). happensAt(walking(fra2gt_4),260).
orientation(fra2gt_3,151,261). coords(fra2gt_3,67,137,261).  visible(fra2gt_3,261). happensAt(walking(fra2gt_3),261). orientation(fra2gt_4,155,261). coords(fra2gt_4,149,147,261).  visible(fra2gt_4,261). happensAt(walking(fra2gt_4),261).
orientation(fra2gt_3,151,262). coords(fra2gt_3,67,136,262).  visible(fra2gt_3,262). happensAt(walking(fra2gt_3),262). orientation(fra2gt_4,155,262). coords(fra2gt_4,148,146,262).  visible(fra2gt_4,262). happensAt(walking(fra2gt_4),262).
orientation(fra2gt_3,151,263). coords(fra2gt_3,68,135,263).  visible(fra2gt_3,263). happensAt(walking(fra2gt_3),263). orientation(fra2gt_4,155,263). coords(fra2gt_4,146,145,263).  visible(fra2gt_4,263). happensAt(walking(fra2gt_4),263).
orientation(fra2gt_3,151,264). coords(fra2gt_3,69,134,264).  visible(fra2gt_3,264). happensAt(walking(fra2gt_3),264). orientation(fra2gt_4,155,264). coords(fra2gt_4,145,144,264).  visible(fra2gt_4,264). happensAt(walking(fra2gt_4),264).
orientation(fra2gt_3,151,265). coords(fra2gt_3,71,134,265).  visible(fra2gt_3,265). happensAt(walking(fra2gt_3),265). orientation(fra2gt_4,155,265). coords(fra2gt_4,145,144,265).  visible(fra2gt_4,265). happensAt(walking(fra2gt_4),265).
orientation(fra2gt_3,151,266). coords(fra2gt_3,72,134,266).  visible(fra2gt_3,266). happensAt(walking(fra2gt_3),266). orientation(fra2gt_4,155,266). coords(fra2gt_4,145,144,266).  visible(fra2gt_4,266). happensAt(walking(fra2gt_4),266).
orientation(fra2gt_3,156,267). coords(fra2gt_3,73,133,267).  visible(fra2gt_3,267). happensAt(walking(fra2gt_3),267). orientation(fra2gt_4,155,267). coords(fra2gt_4,143,143,267).  visible(fra2gt_4,267). happensAt(walking(fra2gt_4),267).
orientation(fra2gt_3,156,268). coords(fra2gt_3,73,133,268).  visible(fra2gt_3,268). happensAt(walking(fra2gt_3),268). orientation(fra2gt_4,155,268). coords(fra2gt_4,142,142,268).  visible(fra2gt_4,268). happensAt(walking(fra2gt_4),268).
orientation(fra2gt_3,156,269). coords(fra2gt_3,73,132,269).  visible(fra2gt_3,269). happensAt(walking(fra2gt_3),269). orientation(fra2gt_4,155,269). coords(fra2gt_4,141,141,269).  visible(fra2gt_4,269). happensAt(walking(fra2gt_4),269).
orientation(fra2gt_3,156,270). coords(fra2gt_3,74,132,270).  visible(fra2gt_3,270). happensAt(walking(fra2gt_3),270). orientation(fra2gt_4,155,270). coords(fra2gt_4,139,141,270).  visible(fra2gt_4,270). happensAt(walking(fra2gt_4),270).
orientation(fra2gt_3,156,271). coords(fra2gt_3,74,131,271).  visible(fra2gt_3,271). happensAt(walking(fra2gt_3),271). orientation(fra2gt_4,155,271). coords(fra2gt_4,137,140,271).  visible(fra2gt_4,271). happensAt(walking(fra2gt_4),271).
orientation(fra2gt_3,156,272). coords(fra2gt_3,75,130,272).  visible(fra2gt_3,272). happensAt(walking(fra2gt_3),272). orientation(fra2gt_4,155,272). coords(fra2gt_4,136,139,272).  visible(fra2gt_4,272). happensAt(walking(fra2gt_4),272).
orientation(fra2gt_3,156,273). coords(fra2gt_3,75,130,273).  visible(fra2gt_3,273). happensAt(walking(fra2gt_3),273). orientation(fra2gt_4,155,273). coords(fra2gt_4,135,137,273).  visible(fra2gt_4,273). happensAt(walking(fra2gt_4),273).
orientation(fra2gt_3,156,274). coords(fra2gt_3,75,129,274).  visible(fra2gt_3,274). happensAt(walking(fra2gt_3),274). orientation(fra2gt_4,155,274). coords(fra2gt_4,135,136,274).  visible(fra2gt_4,274). happensAt(walking(fra2gt_4),274).
#show holdsAt/2.