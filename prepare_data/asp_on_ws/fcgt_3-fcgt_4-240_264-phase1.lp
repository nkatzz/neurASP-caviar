
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

	orientation(fcgt_3,166,240). coords(fcgt_3,101,175,240).  visible(fcgt_3,240). happensAt(walking(fcgt_3),240). orientation(fcgt_4,166,240). coords(fcgt_4,80,205,240).  visible(fcgt_4,240). happensAt(inactive(fcgt_4),240).
orientation(fcgt_3,167,241). coords(fcgt_3,101,175,241).  visible(fcgt_3,241). happensAt(walking(fcgt_3),241). orientation(fcgt_4,166,241). coords(fcgt_4,80,205,241).  visible(fcgt_4,241). happensAt(inactive(fcgt_4),241).
orientation(fcgt_3,166,242). coords(fcgt_3,101,175,242).  visible(fcgt_3,242). happensAt(walking(fcgt_3),242). orientation(fcgt_4,166,242). coords(fcgt_4,80,205,242).  visible(fcgt_4,242). happensAt(inactive(fcgt_4),242).
orientation(fcgt_3,167,243). coords(fcgt_3,101,175,243).  visible(fcgt_3,243). happensAt(walking(fcgt_3),243). orientation(fcgt_4,166,243). coords(fcgt_4,80,205,243).  visible(fcgt_4,243). happensAt(inactive(fcgt_4),243).
orientation(fcgt_3,170,244). coords(fcgt_3,101,175,244).  visible(fcgt_3,244). happensAt(walking(fcgt_3),244). orientation(fcgt_4,166,244). coords(fcgt_4,80,205,244).  visible(fcgt_4,244). happensAt(inactive(fcgt_4),244).
orientation(fcgt_3,170,245). coords(fcgt_3,101,175,245).  visible(fcgt_3,245). happensAt(walking(fcgt_3),245). orientation(fcgt_4,166,245). coords(fcgt_4,80,205,245).  visible(fcgt_4,245). happensAt(inactive(fcgt_4),245).
orientation(fcgt_3,170,246). coords(fcgt_3,101,176,246).  visible(fcgt_3,246). happensAt(walking(fcgt_3),246). orientation(fcgt_4,166,246). coords(fcgt_4,80,205,246).  visible(fcgt_4,246). happensAt(inactive(fcgt_4),246).
orientation(fcgt_3,170,247). coords(fcgt_3,101,176,247).  visible(fcgt_3,247). happensAt(walking(fcgt_3),247). orientation(fcgt_4,166,247). coords(fcgt_4,80,205,247).  visible(fcgt_4,247). happensAt(inactive(fcgt_4),247).
orientation(fcgt_3,170,248). coords(fcgt_3,101,176,248).  visible(fcgt_3,248). happensAt(walking(fcgt_3),248). orientation(fcgt_4,166,248). coords(fcgt_4,80,205,248).  visible(fcgt_4,248). happensAt(inactive(fcgt_4),248).
orientation(fcgt_3,168,249). coords(fcgt_3,101,176,249).  visible(fcgt_3,249). happensAt(walking(fcgt_3),249). orientation(fcgt_4,166,249). coords(fcgt_4,80,205,249).  visible(fcgt_4,249). happensAt(inactive(fcgt_4),249).
orientation(fcgt_3,167,250). coords(fcgt_3,101,176,250).  visible(fcgt_3,250). happensAt(walking(fcgt_3),250). orientation(fcgt_4,166,250). coords(fcgt_4,80,205,250).  visible(fcgt_4,250). happensAt(inactive(fcgt_4),250).
orientation(fcgt_3,165,251). coords(fcgt_3,101,176,251).  visible(fcgt_3,251). happensAt(walking(fcgt_3),251). orientation(fcgt_4,166,251). coords(fcgt_4,80,205,251).  visible(fcgt_4,251). happensAt(inactive(fcgt_4),251).
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
#show holdsAt/2.