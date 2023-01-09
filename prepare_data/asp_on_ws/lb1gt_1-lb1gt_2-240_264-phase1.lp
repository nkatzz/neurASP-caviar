
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

	orientation(lb1gt_1,151,240). coords(lb1gt_1,146,160,240).  visible(lb1gt_1,240). happensAt(walking(lb1gt_1),240). orientation(lb1gt_2,149,240). coords(lb1gt_2,152,124,240).  visible(lb1gt_2,240). happensAt(walking(lb1gt_2),240).
orientation(lb1gt_1,151,241). coords(lb1gt_1,147,160,241).  visible(lb1gt_1,241). happensAt(walking(lb1gt_1),241). orientation(lb1gt_2,149,241). coords(lb1gt_2,153,124,241).  visible(lb1gt_2,241). happensAt(walking(lb1gt_2),241).
orientation(lb1gt_1,151,242). coords(lb1gt_1,147,160,242).  visible(lb1gt_1,242). happensAt(walking(lb1gt_1),242). orientation(lb1gt_2,149,242). coords(lb1gt_2,153,124,242).  visible(lb1gt_2,242). happensAt(walking(lb1gt_2),242).
orientation(lb1gt_1,151,243). coords(lb1gt_1,147,160,243).  visible(lb1gt_1,243). happensAt(walking(lb1gt_1),243). orientation(lb1gt_2,149,243). coords(lb1gt_2,154,125,243).  visible(lb1gt_2,243). happensAt(walking(lb1gt_2),243).
orientation(lb1gt_1,151,244). coords(lb1gt_1,147,160,244).  visible(lb1gt_1,244). happensAt(walking(lb1gt_1),244). orientation(lb1gt_2,149,244). coords(lb1gt_2,154,125,244).  visible(lb1gt_2,244). happensAt(walking(lb1gt_2),244).
orientation(lb1gt_1,151,245). coords(lb1gt_1,147,160,245).  visible(lb1gt_1,245). happensAt(walking(lb1gt_1),245). orientation(lb1gt_2,149,245). coords(lb1gt_2,154,126,245).  visible(lb1gt_2,245). happensAt(walking(lb1gt_2),245).
orientation(lb1gt_1,151,246). coords(lb1gt_1,147,160,246).  visible(lb1gt_1,246). happensAt(walking(lb1gt_1),246). orientation(lb1gt_2,149,246). coords(lb1gt_2,156,127,246).  visible(lb1gt_2,246). happensAt(walking(lb1gt_2),246).
orientation(lb1gt_1,151,247). coords(lb1gt_1,148,160,247).  visible(lb1gt_1,247). happensAt(walking(lb1gt_1),247). orientation(lb1gt_2,149,247). coords(lb1gt_2,157,128,247).  visible(lb1gt_2,247). happensAt(walking(lb1gt_2),247).
orientation(lb1gt_1,151,248). coords(lb1gt_1,148,160,248).  visible(lb1gt_1,248). happensAt(walking(lb1gt_1),248). orientation(lb1gt_2,149,248). coords(lb1gt_2,158,129,248).  visible(lb1gt_2,248). happensAt(walking(lb1gt_2),248).
orientation(lb1gt_1,151,249). coords(lb1gt_1,148,160,249).  visible(lb1gt_1,249). happensAt(walking(lb1gt_1),249). orientation(lb1gt_2,149,249). coords(lb1gt_2,159,130,249).  visible(lb1gt_2,249). happensAt(walking(lb1gt_2),249).
orientation(lb1gt_1,151,250). coords(lb1gt_1,148,160,250).  visible(lb1gt_1,250). happensAt(walking(lb1gt_1),250). orientation(lb1gt_2,149,250). coords(lb1gt_2,159,130,250).  visible(lb1gt_2,250). happensAt(walking(lb1gt_2),250).
orientation(lb1gt_1,151,251). coords(lb1gt_1,149,160,251).  visible(lb1gt_1,251). happensAt(walking(lb1gt_1),251). orientation(lb1gt_2,149,251). coords(lb1gt_2,160,130,251).  visible(lb1gt_2,251). happensAt(walking(lb1gt_2),251).
orientation(lb1gt_1,151,252). coords(lb1gt_1,150,160,252).  visible(lb1gt_1,252). happensAt(walking(lb1gt_1),252). orientation(lb1gt_2,149,252). coords(lb1gt_2,161,130,252).  visible(lb1gt_2,252). happensAt(walking(lb1gt_2),252).
orientation(lb1gt_1,151,253). coords(lb1gt_1,151,160,253).  visible(lb1gt_1,253). happensAt(active(lb1gt_1),253). orientation(lb1gt_2,149,253). coords(lb1gt_2,161,131,253).  visible(lb1gt_2,253). happensAt(walking(lb1gt_2),253).
orientation(lb1gt_1,151,254). coords(lb1gt_1,151,160,254).  visible(lb1gt_1,254). happensAt(active(lb1gt_1),254). orientation(lb1gt_2,149,254). coords(lb1gt_2,161,131,254).  visible(lb1gt_2,254). happensAt(walking(lb1gt_2),254).
orientation(lb1gt_1,151,255). coords(lb1gt_1,151,160,255).  visible(lb1gt_1,255). happensAt(active(lb1gt_1),255). orientation(lb1gt_2,149,255). coords(lb1gt_2,161,131,255).  visible(lb1gt_2,255). happensAt(walking(lb1gt_2),255).
orientation(lb1gt_1,151,256). coords(lb1gt_1,152,160,256).  visible(lb1gt_1,256). happensAt(active(lb1gt_1),256). orientation(lb1gt_2,149,256). coords(lb1gt_2,161,131,256).  visible(lb1gt_2,256). happensAt(walking(lb1gt_2),256).
orientation(lb1gt_1,151,257). coords(lb1gt_1,152,160,257).  visible(lb1gt_1,257). happensAt(active(lb1gt_1),257). orientation(lb1gt_2,149,257). coords(lb1gt_2,161,131,257).  visible(lb1gt_2,257). happensAt(walking(lb1gt_2),257).
orientation(lb1gt_1,151,258). coords(lb1gt_1,152,160,258).  visible(lb1gt_1,258). happensAt(active(lb1gt_1),258). orientation(lb1gt_2,149,258). coords(lb1gt_2,161,131,258).  visible(lb1gt_2,258). happensAt(walking(lb1gt_2),258).
orientation(lb1gt_1,151,259). coords(lb1gt_1,152,160,259).  visible(lb1gt_1,259). happensAt(active(lb1gt_1),259). orientation(lb1gt_2,149,259). coords(lb1gt_2,162,132,259).  visible(lb1gt_2,259). happensAt(walking(lb1gt_2),259).
orientation(lb1gt_1,151,260). coords(lb1gt_1,152,160,260).  visible(lb1gt_1,260). happensAt(active(lb1gt_1),260). orientation(lb1gt_2,149,260). coords(lb1gt_2,163,132,260).  visible(lb1gt_2,260). happensAt(walking(lb1gt_2),260).
orientation(lb1gt_1,151,261). coords(lb1gt_1,153,160,261).  visible(lb1gt_1,261). happensAt(active(lb1gt_1),261). orientation(lb1gt_2,149,261). coords(lb1gt_2,165,132,261).  visible(lb1gt_2,261). happensAt(walking(lb1gt_2),261).
orientation(lb1gt_1,151,262). coords(lb1gt_1,153,160,262).  visible(lb1gt_1,262). happensAt(active(lb1gt_1),262). orientation(lb1gt_2,149,262). coords(lb1gt_2,168,132,262).  visible(lb1gt_2,262). happensAt(walking(lb1gt_2),262).
orientation(lb1gt_1,151,263). coords(lb1gt_1,153,160,263).  visible(lb1gt_1,263). happensAt(active(lb1gt_1),263). orientation(lb1gt_2,149,263). coords(lb1gt_2,170,132,263).  visible(lb1gt_2,263). happensAt(walking(lb1gt_2),263).
orientation(lb1gt_1,151,264). coords(lb1gt_1,153,160,264).  visible(lb1gt_1,264). happensAt(active(lb1gt_1),264). orientation(lb1gt_2,149,264). coords(lb1gt_2,172,134,264).  visible(lb1gt_2,264). happensAt(walking(lb1gt_2),264).
#show holdsAt/2.