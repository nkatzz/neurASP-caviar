
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

	orientation(mws1gt_1,155,154). coords(mws1gt_1,140,148,154).  visible(mws1gt_1,154). happensAt(walking(mws1gt_1),154). orientation(mws1gt_5,145,154). coords(mws1gt_5,71,81,154).  visible(mws1gt_5,154). happensAt(walking(mws1gt_5),154).
orientation(mws1gt_1,155,155). coords(mws1gt_1,140,147,155).  visible(mws1gt_1,155). happensAt(walking(mws1gt_1),155). orientation(mws1gt_5,145,155). coords(mws1gt_5,71,82,155).  visible(mws1gt_5,155). happensAt(walking(mws1gt_5),155).
orientation(mws1gt_1,155,156). coords(mws1gt_1,141,147,156).  visible(mws1gt_1,156). happensAt(walking(mws1gt_1),156). orientation(mws1gt_5,145,156). coords(mws1gt_5,71,82,156).  visible(mws1gt_5,156). happensAt(walking(mws1gt_5),156).
orientation(mws1gt_1,155,157). coords(mws1gt_1,141,146,157).  visible(mws1gt_1,157). happensAt(walking(mws1gt_1),157). orientation(mws1gt_5,145,157). coords(mws1gt_5,71,82,157).  visible(mws1gt_5,157). happensAt(walking(mws1gt_5),157).
orientation(mws1gt_1,155,158). coords(mws1gt_1,141,145,158).  visible(mws1gt_1,158). happensAt(walking(mws1gt_1),158). orientation(mws1gt_5,145,158). coords(mws1gt_5,71,83,158).  visible(mws1gt_5,158). happensAt(walking(mws1gt_5),158).
orientation(mws1gt_1,155,159). coords(mws1gt_1,141,144,159).  visible(mws1gt_1,159). happensAt(walking(mws1gt_1),159). orientation(mws1gt_5,145,159). coords(mws1gt_5,71,83,159).  visible(mws1gt_5,159). happensAt(walking(mws1gt_5),159).
orientation(mws1gt_1,155,160). coords(mws1gt_1,142,143,160).  visible(mws1gt_1,160). happensAt(walking(mws1gt_1),160). orientation(mws1gt_5,145,160). coords(mws1gt_5,70,83,160).  visible(mws1gt_5,160). happensAt(walking(mws1gt_5),160).
orientation(mws1gt_1,155,161). coords(mws1gt_1,142,143,161).  visible(mws1gt_1,161). happensAt(walking(mws1gt_1),161). orientation(mws1gt_5,145,161). coords(mws1gt_5,68,83,161).  visible(mws1gt_5,161). happensAt(walking(mws1gt_5),161).
orientation(mws1gt_1,155,162). coords(mws1gt_1,142,143,162).  visible(mws1gt_1,162). happensAt(walking(mws1gt_1),162). orientation(mws1gt_5,145,162). coords(mws1gt_5,67,83,162).  visible(mws1gt_5,162). happensAt(walking(mws1gt_5),162).
orientation(mws1gt_1,155,163). coords(mws1gt_1,142,143,163).  visible(mws1gt_1,163). happensAt(walking(mws1gt_1),163). orientation(mws1gt_5,145,163). coords(mws1gt_5,67,83,163).  visible(mws1gt_5,163). happensAt(walking(mws1gt_5),163).
orientation(mws1gt_1,155,164). coords(mws1gt_1,142,143,164).  visible(mws1gt_1,164). happensAt(walking(mws1gt_1),164). orientation(mws1gt_5,145,164). coords(mws1gt_5,66,84,164).  visible(mws1gt_5,164). happensAt(walking(mws1gt_5),164).
orientation(mws1gt_1,155,165). coords(mws1gt_1,142,143,165).  visible(mws1gt_1,165). happensAt(walking(mws1gt_1),165). orientation(mws1gt_5,145,165). coords(mws1gt_5,65,86,165).  visible(mws1gt_5,165). happensAt(walking(mws1gt_5),165).
orientation(mws1gt_1,155,166). coords(mws1gt_1,142,143,166).  visible(mws1gt_1,166). happensAt(walking(mws1gt_1),166). orientation(mws1gt_5,145,166). coords(mws1gt_5,64,87,166).  visible(mws1gt_5,166). happensAt(walking(mws1gt_5),166).
orientation(mws1gt_1,155,167). coords(mws1gt_1,142,143,167).  visible(mws1gt_1,167). happensAt(walking(mws1gt_1),167). orientation(mws1gt_5,145,167). coords(mws1gt_5,64,88,167).  visible(mws1gt_5,167). happensAt(walking(mws1gt_5),167).
orientation(mws1gt_1,155,168). coords(mws1gt_1,148,142,168).  visible(mws1gt_1,168). happensAt(walking(mws1gt_1),168). orientation(mws1gt_5,145,168). coords(mws1gt_5,64,89,168).  visible(mws1gt_5,168). happensAt(walking(mws1gt_5),168).
orientation(mws1gt_1,155,169). coords(mws1gt_1,148,142,169).  visible(mws1gt_1,169). happensAt(walking(mws1gt_1),169). orientation(mws1gt_5,145,169). coords(mws1gt_5,64,90,169).  visible(mws1gt_5,169). happensAt(walking(mws1gt_5),169).
orientation(mws1gt_1,155,170). coords(mws1gt_1,148,142,170).  visible(mws1gt_1,170). happensAt(walking(mws1gt_1),170). orientation(mws1gt_5,145,170). coords(mws1gt_5,64,90,170).  visible(mws1gt_5,170). happensAt(walking(mws1gt_5),170).
orientation(mws1gt_1,155,171). coords(mws1gt_1,149,141,171).  visible(mws1gt_1,171). happensAt(walking(mws1gt_1),171). orientation(mws1gt_5,145,171). coords(mws1gt_5,63,90,171).  visible(mws1gt_5,171). happensAt(walking(mws1gt_5),171).
orientation(mws1gt_1,155,172). coords(mws1gt_1,149,140,172).  visible(mws1gt_1,172). happensAt(walking(mws1gt_1),172). orientation(mws1gt_5,145,172). coords(mws1gt_5,63,90,172).  visible(mws1gt_5,172). happensAt(walking(mws1gt_5),172).
orientation(mws1gt_1,155,173). coords(mws1gt_1,149,140,173).  visible(mws1gt_1,173). happensAt(walking(mws1gt_1),173). orientation(mws1gt_5,145,173). coords(mws1gt_5,62,90,173).  visible(mws1gt_5,173). happensAt(walking(mws1gt_5),173).
orientation(mws1gt_1,155,174). coords(mws1gt_1,150,139,174).  visible(mws1gt_1,174). happensAt(walking(mws1gt_1),174). orientation(mws1gt_5,145,174). coords(mws1gt_5,62,90,174).  visible(mws1gt_5,174). happensAt(walking(mws1gt_5),174).
orientation(mws1gt_1,155,175). coords(mws1gt_1,150,139,175).  visible(mws1gt_1,175). happensAt(walking(mws1gt_1),175). orientation(mws1gt_5,145,175). coords(mws1gt_5,61,90,175).  visible(mws1gt_5,175). happensAt(walking(mws1gt_5),175).
orientation(mws1gt_1,155,176). coords(mws1gt_1,150,138,176).  visible(mws1gt_1,176). happensAt(walking(mws1gt_1),176). orientation(mws1gt_5,145,176). coords(mws1gt_5,61,91,176).  visible(mws1gt_5,176). happensAt(walking(mws1gt_5),176).
orientation(mws1gt_1,155,177). coords(mws1gt_1,151,137,177).  visible(mws1gt_1,177). happensAt(walking(mws1gt_1),177). orientation(mws1gt_5,145,177). coords(mws1gt_5,60,91,177).  visible(mws1gt_5,177). happensAt(walking(mws1gt_5),177).
orientation(mws1gt_1,155,178). coords(mws1gt_1,151,136,178).  visible(mws1gt_1,178). happensAt(walking(mws1gt_1),178). orientation(mws1gt_5,145,178). coords(mws1gt_5,59,93,178).  visible(mws1gt_5,178). happensAt(walking(mws1gt_5),178).
#show holdsAt/2.