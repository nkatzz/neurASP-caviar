
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

	orientation(wk1gt_1,35,248). coords(wk1gt_1,235,278,248).  visible(wk1gt_1,248). happensAt(walking(wk1gt_1),248). orientation(wk1gt_4,163,248). coords(wk1gt_4,47,181,248).  visible(wk1gt_4,248). happensAt(inactive(wk1gt_4),248).
orientation(wk1gt_1,35,249). coords(wk1gt_1,231,275,249).  visible(wk1gt_1,249). happensAt(walking(wk1gt_1),249). orientation(wk1gt_4,161,249). coords(wk1gt_4,47,181,249).  visible(wk1gt_4,249). happensAt(inactive(wk1gt_4),249).
orientation(wk1gt_1,35,250). coords(wk1gt_1,228,273,250).  visible(wk1gt_1,250). happensAt(walking(wk1gt_1),250). orientation(wk1gt_4,160,250). coords(wk1gt_4,47,181,250).  visible(wk1gt_4,250). happensAt(inactive(wk1gt_4),250).
orientation(wk1gt_1,35,251). coords(wk1gt_1,228,272,251).  visible(wk1gt_1,251). happensAt(walking(wk1gt_1),251). orientation(wk1gt_4,162,251). coords(wk1gt_4,47,181,251).  visible(wk1gt_4,251). happensAt(inactive(wk1gt_4),251).
orientation(wk1gt_1,26,252). coords(wk1gt_1,228,272,252).  visible(wk1gt_1,252). happensAt(walking(wk1gt_1),252). orientation(wk1gt_4,161,252). coords(wk1gt_4,47,181,252).  visible(wk1gt_4,252). happensAt(inactive(wk1gt_4),252).
orientation(wk1gt_1,21,253). coords(wk1gt_1,228,272,253).  visible(wk1gt_1,253). happensAt(walking(wk1gt_1),253). orientation(wk1gt_4,163,253). coords(wk1gt_4,47,181,253).  visible(wk1gt_4,253). happensAt(active(wk1gt_4),253).
orientation(wk1gt_1,21,254). coords(wk1gt_1,228,272,254).  visible(wk1gt_1,254). happensAt(walking(wk1gt_1),254). orientation(wk1gt_4,161,254). coords(wk1gt_4,47,181,254).  visible(wk1gt_4,254). happensAt(active(wk1gt_4),254).
orientation(wk1gt_1,11,255). coords(wk1gt_1,227,272,255).  visible(wk1gt_1,255). happensAt(walking(wk1gt_1),255). orientation(wk1gt_4,163,255). coords(wk1gt_4,47,181,255).  visible(wk1gt_4,255). happensAt(active(wk1gt_4),255).
orientation(wk1gt_1,11,256). coords(wk1gt_1,226,270,256).  visible(wk1gt_1,256). happensAt(walking(wk1gt_1),256). orientation(wk1gt_4,161,256). coords(wk1gt_4,47,181,256).  visible(wk1gt_4,256). happensAt(active(wk1gt_4),256).
orientation(wk1gt_1,11,257). coords(wk1gt_1,225,269,257).  visible(wk1gt_1,257). happensAt(walking(wk1gt_1),257). orientation(wk1gt_4,160,257). coords(wk1gt_4,47,181,257).  visible(wk1gt_4,257). happensAt(active(wk1gt_4),257).
orientation(wk1gt_1,11,258). coords(wk1gt_1,225,267,258).  visible(wk1gt_1,258). happensAt(walking(wk1gt_1),258). orientation(wk1gt_4,157,258). coords(wk1gt_4,47,181,258).  visible(wk1gt_4,258). happensAt(active(wk1gt_4),258).
orientation(wk1gt_1,11,259). coords(wk1gt_1,224,267,259).  visible(wk1gt_1,259). happensAt(walking(wk1gt_1),259). orientation(wk1gt_4,160,259). coords(wk1gt_4,47,180,259).  visible(wk1gt_4,259). happensAt(active(wk1gt_4),259).
orientation(wk1gt_1,11,260). coords(wk1gt_1,224,264,260).  visible(wk1gt_1,260). happensAt(walking(wk1gt_1),260). orientation(wk1gt_4,160,260). coords(wk1gt_4,48,180,260).  visible(wk1gt_4,260). happensAt(active(wk1gt_4),260).
orientation(wk1gt_1,17,261). coords(wk1gt_1,224,260,261).  visible(wk1gt_1,261). happensAt(walking(wk1gt_1),261). orientation(wk1gt_4,159,261). coords(wk1gt_4,48,179,261).  visible(wk1gt_4,261). happensAt(active(wk1gt_4),261).
orientation(wk1gt_1,22,262). coords(wk1gt_1,224,256,262).  visible(wk1gt_1,262). happensAt(walking(wk1gt_1),262). orientation(wk1gt_4,160,262). coords(wk1gt_4,48,179,262).  visible(wk1gt_4,262). happensAt(active(wk1gt_4),262).
orientation(wk1gt_1,22,263). coords(wk1gt_1,223,254,263).  visible(wk1gt_1,263). happensAt(walking(wk1gt_1),263). orientation(wk1gt_4,158,263). coords(wk1gt_4,48,179,263).  visible(wk1gt_4,263). happensAt(active(wk1gt_4),263).
orientation(wk1gt_1,22,264). coords(wk1gt_1,223,252,264).  visible(wk1gt_1,264). happensAt(walking(wk1gt_1),264). orientation(wk1gt_4,162,264). coords(wk1gt_4,48,179,264).  visible(wk1gt_4,264). happensAt(active(wk1gt_4),264).
orientation(wk1gt_1,22,265). coords(wk1gt_1,222,251,265).  visible(wk1gt_1,265). happensAt(walking(wk1gt_1),265). orientation(wk1gt_4,163,265). coords(wk1gt_4,48,179,265).  visible(wk1gt_4,265). happensAt(active(wk1gt_4),265).
orientation(wk1gt_1,22,266). coords(wk1gt_1,222,251,266).  visible(wk1gt_1,266). happensAt(walking(wk1gt_1),266). orientation(wk1gt_4,154,266). coords(wk1gt_4,48,179,266).  visible(wk1gt_4,266). happensAt(active(wk1gt_4),266).
orientation(wk1gt_1,8,267). coords(wk1gt_1,221,250,267).  visible(wk1gt_1,267). happensAt(walking(wk1gt_1),267). orientation(wk1gt_4,145,267). coords(wk1gt_4,48,179,267).  visible(wk1gt_4,267). happensAt(active(wk1gt_4),267).
orientation(wk1gt_1,8,268). coords(wk1gt_1,218,249,268).  visible(wk1gt_1,268). happensAt(walking(wk1gt_1),268). orientation(wk1gt_4,152,268). coords(wk1gt_4,48,179,268).  visible(wk1gt_4,268). happensAt(active(wk1gt_4),268).
orientation(wk1gt_1,8,269). coords(wk1gt_1,218,247,269).  visible(wk1gt_1,269). happensAt(walking(wk1gt_1),269). orientation(wk1gt_4,156,269). coords(wk1gt_4,48,178,269).  visible(wk1gt_4,269). happensAt(active(wk1gt_4),269).
orientation(wk1gt_1,3,270). coords(wk1gt_1,216,245,270).  visible(wk1gt_1,270). happensAt(walking(wk1gt_1),270). orientation(wk1gt_4,158,270). coords(wk1gt_4,48,178,270).  visible(wk1gt_4,270). happensAt(active(wk1gt_4),270).
orientation(wk1gt_1,3,271). coords(wk1gt_1,216,242,271).  visible(wk1gt_1,271). happensAt(walking(wk1gt_1),271). orientation(wk1gt_4,157,271). coords(wk1gt_4,49,178,271).  visible(wk1gt_4,271). happensAt(active(wk1gt_4),271).
orientation(wk1gt_1,3,272). coords(wk1gt_1,213,241,272).  visible(wk1gt_1,272). happensAt(walking(wk1gt_1),272). orientation(wk1gt_4,153,272). coords(wk1gt_4,49,178,272).  visible(wk1gt_4,272). happensAt(active(wk1gt_4),272).
#show holdsAt/2.