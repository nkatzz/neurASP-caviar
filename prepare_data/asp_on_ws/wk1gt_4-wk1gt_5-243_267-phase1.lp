
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

	orientation(wk1gt_4,160,243). coords(wk1gt_4,47,181,243).  visible(wk1gt_4,243). happensAt(inactive(wk1gt_4),243). orientation(wk1gt_5,162,243). coords(wk1gt_5,50,166,243).  visible(wk1gt_5,243). happensAt(inactive(wk1gt_5),243).
orientation(wk1gt_4,162,244). coords(wk1gt_4,47,181,244).  visible(wk1gt_4,244). happensAt(inactive(wk1gt_4),244). orientation(wk1gt_5,162,244). coords(wk1gt_5,50,166,244).  visible(wk1gt_5,244). happensAt(inactive(wk1gt_5),244).
orientation(wk1gt_4,161,245). coords(wk1gt_4,47,181,245).  visible(wk1gt_4,245). happensAt(inactive(wk1gt_4),245). orientation(wk1gt_5,162,245). coords(wk1gt_5,50,166,245).  visible(wk1gt_5,245). happensAt(inactive(wk1gt_5),245).
orientation(wk1gt_4,155,246). coords(wk1gt_4,47,181,246).  visible(wk1gt_4,246). happensAt(inactive(wk1gt_4),246). orientation(wk1gt_5,162,246). coords(wk1gt_5,50,166,246).  visible(wk1gt_5,246). happensAt(inactive(wk1gt_5),246).
orientation(wk1gt_4,154,247). coords(wk1gt_4,47,181,247).  visible(wk1gt_4,247). happensAt(inactive(wk1gt_4),247). orientation(wk1gt_5,162,247). coords(wk1gt_5,50,166,247).  visible(wk1gt_5,247). happensAt(inactive(wk1gt_5),247).
orientation(wk1gt_4,163,248). coords(wk1gt_4,47,181,248).  visible(wk1gt_4,248). happensAt(inactive(wk1gt_4),248). orientation(wk1gt_5,162,248). coords(wk1gt_5,50,166,248).  visible(wk1gt_5,248). happensAt(inactive(wk1gt_5),248).
orientation(wk1gt_4,161,249). coords(wk1gt_4,47,181,249).  visible(wk1gt_4,249). happensAt(inactive(wk1gt_4),249). orientation(wk1gt_5,162,249). coords(wk1gt_5,50,166,249).  visible(wk1gt_5,249). happensAt(inactive(wk1gt_5),249).
orientation(wk1gt_4,160,250). coords(wk1gt_4,47,181,250).  visible(wk1gt_4,250). happensAt(inactive(wk1gt_4),250). orientation(wk1gt_5,162,250). coords(wk1gt_5,50,166,250).  visible(wk1gt_5,250). happensAt(inactive(wk1gt_5),250).
orientation(wk1gt_4,162,251). coords(wk1gt_4,47,181,251).  visible(wk1gt_4,251). happensAt(inactive(wk1gt_4),251). orientation(wk1gt_5,162,251). coords(wk1gt_5,50,166,251).  visible(wk1gt_5,251). happensAt(inactive(wk1gt_5),251).
orientation(wk1gt_4,161,252). coords(wk1gt_4,47,181,252).  visible(wk1gt_4,252). happensAt(inactive(wk1gt_4),252). orientation(wk1gt_5,162,252). coords(wk1gt_5,50,166,252).  visible(wk1gt_5,252). happensAt(inactive(wk1gt_5),252).
orientation(wk1gt_4,163,253). coords(wk1gt_4,47,181,253).  visible(wk1gt_4,253). happensAt(active(wk1gt_4),253). orientation(wk1gt_5,162,253). coords(wk1gt_5,50,166,253).  visible(wk1gt_5,253). happensAt(inactive(wk1gt_5),253).
orientation(wk1gt_4,161,254). coords(wk1gt_4,47,181,254).  visible(wk1gt_4,254). happensAt(active(wk1gt_4),254). orientation(wk1gt_5,162,254). coords(wk1gt_5,50,166,254).  visible(wk1gt_5,254). happensAt(inactive(wk1gt_5),254).
orientation(wk1gt_4,163,255). coords(wk1gt_4,47,181,255).  visible(wk1gt_4,255). happensAt(active(wk1gt_4),255). orientation(wk1gt_5,162,255). coords(wk1gt_5,50,166,255).  visible(wk1gt_5,255). happensAt(inactive(wk1gt_5),255).
orientation(wk1gt_4,161,256). coords(wk1gt_4,47,181,256).  visible(wk1gt_4,256). happensAt(active(wk1gt_4),256). orientation(wk1gt_5,162,256). coords(wk1gt_5,50,166,256).  visible(wk1gt_5,256). happensAt(inactive(wk1gt_5),256).
orientation(wk1gt_4,160,257). coords(wk1gt_4,47,181,257).  visible(wk1gt_4,257). happensAt(active(wk1gt_4),257). orientation(wk1gt_5,162,257). coords(wk1gt_5,50,166,257).  visible(wk1gt_5,257). happensAt(inactive(wk1gt_5),257).
orientation(wk1gt_4,157,258). coords(wk1gt_4,47,181,258).  visible(wk1gt_4,258). happensAt(active(wk1gt_4),258). orientation(wk1gt_5,162,258). coords(wk1gt_5,50,166,258).  visible(wk1gt_5,258). happensAt(inactive(wk1gt_5),258).
orientation(wk1gt_4,160,259). coords(wk1gt_4,47,180,259).  visible(wk1gt_4,259). happensAt(active(wk1gt_4),259). orientation(wk1gt_5,162,259). coords(wk1gt_5,50,166,259).  visible(wk1gt_5,259). happensAt(inactive(wk1gt_5),259).
orientation(wk1gt_4,160,260). coords(wk1gt_4,48,180,260).  visible(wk1gt_4,260). happensAt(active(wk1gt_4),260). orientation(wk1gt_5,162,260). coords(wk1gt_5,50,166,260).  visible(wk1gt_5,260). happensAt(inactive(wk1gt_5),260).
orientation(wk1gt_4,159,261). coords(wk1gt_4,48,179,261).  visible(wk1gt_4,261). happensAt(active(wk1gt_4),261). orientation(wk1gt_5,162,261). coords(wk1gt_5,50,166,261).  visible(wk1gt_5,261). happensAt(inactive(wk1gt_5),261).
orientation(wk1gt_4,160,262). coords(wk1gt_4,48,179,262).  visible(wk1gt_4,262). happensAt(active(wk1gt_4),262). orientation(wk1gt_5,162,262). coords(wk1gt_5,50,166,262).  visible(wk1gt_5,262). happensAt(inactive(wk1gt_5),262).
orientation(wk1gt_4,158,263). coords(wk1gt_4,48,179,263).  visible(wk1gt_4,263). happensAt(active(wk1gt_4),263). orientation(wk1gt_5,162,263). coords(wk1gt_5,50,166,263).  visible(wk1gt_5,263). happensAt(inactive(wk1gt_5),263).
orientation(wk1gt_4,162,264). coords(wk1gt_4,48,179,264).  visible(wk1gt_4,264). happensAt(active(wk1gt_4),264). orientation(wk1gt_5,162,264). coords(wk1gt_5,50,166,264).  visible(wk1gt_5,264). happensAt(inactive(wk1gt_5),264).
orientation(wk1gt_4,163,265). coords(wk1gt_4,48,179,265).  visible(wk1gt_4,265). happensAt(active(wk1gt_4),265). orientation(wk1gt_5,162,265). coords(wk1gt_5,50,166,265).  visible(wk1gt_5,265). happensAt(inactive(wk1gt_5),265).
orientation(wk1gt_4,154,266). coords(wk1gt_4,48,179,266).  visible(wk1gt_4,266). happensAt(active(wk1gt_4),266). orientation(wk1gt_5,162,266). coords(wk1gt_5,50,166,266).  visible(wk1gt_5,266). happensAt(inactive(wk1gt_5),266).
orientation(wk1gt_4,145,267). coords(wk1gt_4,48,179,267).  visible(wk1gt_4,267). happensAt(active(wk1gt_4),267). orientation(wk1gt_5,162,267). coords(wk1gt_5,50,166,267).  visible(wk1gt_5,267). happensAt(inactive(wk1gt_5),267).
#show holdsAt/2.