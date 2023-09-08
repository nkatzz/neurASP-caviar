
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

	orientation(wk2gt_1,126,240). coords(wk2gt_1,115,53,240).  visible(wk2gt_1,240). happensAt(inactive(wk2gt_1),240). orientation(wk2gt_2,141,240). coords(wk2gt_2,98,62,240).  visible(wk2gt_2,240). happensAt(inactive(wk2gt_2),240).
orientation(wk2gt_1,126,241). coords(wk2gt_1,115,53,241).  visible(wk2gt_1,241). happensAt(inactive(wk2gt_1),241). orientation(wk2gt_2,141,241). coords(wk2gt_2,98,62,241).  visible(wk2gt_2,241). happensAt(inactive(wk2gt_2),241).
orientation(wk2gt_1,126,242). coords(wk2gt_1,115,53,242).  visible(wk2gt_1,242). happensAt(inactive(wk2gt_1),242). orientation(wk2gt_2,141,242). coords(wk2gt_2,98,62,242).  visible(wk2gt_2,242). happensAt(inactive(wk2gt_2),242).
orientation(wk2gt_1,126,243). coords(wk2gt_1,115,53,243).  visible(wk2gt_1,243). happensAt(inactive(wk2gt_1),243). orientation(wk2gt_2,141,243). coords(wk2gt_2,98,62,243).  visible(wk2gt_2,243). happensAt(inactive(wk2gt_2),243).
orientation(wk2gt_1,126,244). coords(wk2gt_1,115,53,244).  visible(wk2gt_1,244). happensAt(inactive(wk2gt_1),244). orientation(wk2gt_2,141,244). coords(wk2gt_2,98,62,244).  visible(wk2gt_2,244). happensAt(inactive(wk2gt_2),244).
orientation(wk2gt_1,126,245). coords(wk2gt_1,115,53,245).  visible(wk2gt_1,245). happensAt(inactive(wk2gt_1),245). orientation(wk2gt_2,141,245). coords(wk2gt_2,98,62,245).  visible(wk2gt_2,245). happensAt(inactive(wk2gt_2),245).
orientation(wk2gt_1,126,246). coords(wk2gt_1,115,53,246).  visible(wk2gt_1,246). happensAt(inactive(wk2gt_1),246). orientation(wk2gt_2,141,246). coords(wk2gt_2,98,62,246).  visible(wk2gt_2,246). happensAt(inactive(wk2gt_2),246).
orientation(wk2gt_1,126,247). coords(wk2gt_1,115,53,247).  visible(wk2gt_1,247). happensAt(inactive(wk2gt_1),247). orientation(wk2gt_2,141,247). coords(wk2gt_2,98,62,247).  visible(wk2gt_2,247). happensAt(inactive(wk2gt_2),247).
orientation(wk2gt_1,126,248). coords(wk2gt_1,115,53,248).  visible(wk2gt_1,248). happensAt(inactive(wk2gt_1),248). orientation(wk2gt_2,141,248). coords(wk2gt_2,98,62,248).  visible(wk2gt_2,248). happensAt(inactive(wk2gt_2),248).
orientation(wk2gt_1,126,249). coords(wk2gt_1,115,53,249).  visible(wk2gt_1,249). happensAt(inactive(wk2gt_1),249). orientation(wk2gt_2,141,249). coords(wk2gt_2,98,62,249).  visible(wk2gt_2,249). happensAt(inactive(wk2gt_2),249).
orientation(wk2gt_1,125,250). coords(wk2gt_1,115,53,250).  visible(wk2gt_1,250). happensAt(inactive(wk2gt_1),250). orientation(wk2gt_2,141,250). coords(wk2gt_2,98,62,250).  visible(wk2gt_2,250). happensAt(inactive(wk2gt_2),250).
orientation(wk2gt_1,125,251). coords(wk2gt_1,115,53,251).  visible(wk2gt_1,251). happensAt(inactive(wk2gt_1),251). orientation(wk2gt_2,141,251). coords(wk2gt_2,98,62,251).  visible(wk2gt_2,251). happensAt(inactive(wk2gt_2),251).
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
#show holdsAt/2.