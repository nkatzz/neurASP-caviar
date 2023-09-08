
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

	orientation(fra2gt_3,165,151). coords(fra2gt_3,34,204,151).  visible(fra2gt_3,151). happensAt(walking(fra2gt_3),151). orientation(fra2gt_5,150,151). coords(fra2gt_5,29,120,151).  visible(fra2gt_5,151). happensAt(walking(fra2gt_5),151).
orientation(fra2gt_3,165,152). coords(fra2gt_3,34,203,152).  visible(fra2gt_3,152). happensAt(walking(fra2gt_3),152). orientation(fra2gt_5,150,152). coords(fra2gt_5,29,120,152).  visible(fra2gt_5,152). happensAt(walking(fra2gt_5),152).
orientation(fra2gt_3,165,153). coords(fra2gt_3,34,203,153).  visible(fra2gt_3,153). happensAt(walking(fra2gt_3),153). orientation(fra2gt_5,150,153). coords(fra2gt_5,29,120,153).  visible(fra2gt_5,153). happensAt(walking(fra2gt_5),153).
orientation(fra2gt_3,165,154). coords(fra2gt_3,34,202,154).  visible(fra2gt_3,154). happensAt(walking(fra2gt_3),154). orientation(fra2gt_5,150,154). coords(fra2gt_5,29,120,154).  visible(fra2gt_5,154). happensAt(walking(fra2gt_5),154).
orientation(fra2gt_3,165,155). coords(fra2gt_3,34,202,155).  visible(fra2gt_3,155). happensAt(walking(fra2gt_3),155). orientation(fra2gt_5,150,155). coords(fra2gt_5,29,120,155).  visible(fra2gt_5,155). happensAt(walking(fra2gt_5),155).
orientation(fra2gt_3,165,156). coords(fra2gt_3,34,201,156).  visible(fra2gt_3,156). happensAt(walking(fra2gt_3),156). orientation(fra2gt_5,150,156). coords(fra2gt_5,29,120,156).  visible(fra2gt_5,156). happensAt(walking(fra2gt_5),156).
orientation(fra2gt_3,165,157). coords(fra2gt_3,34,201,157).  visible(fra2gt_3,157). happensAt(walking(fra2gt_3),157). orientation(fra2gt_5,150,157). coords(fra2gt_5,29,118,157).  visible(fra2gt_5,157). happensAt(walking(fra2gt_5),157).
orientation(fra2gt_3,165,158). coords(fra2gt_3,34,201,158).  visible(fra2gt_3,158). happensAt(walking(fra2gt_3),158). orientation(fra2gt_5,150,158). coords(fra2gt_5,29,118,158).  visible(fra2gt_5,158). happensAt(walking(fra2gt_5),158).
orientation(fra2gt_3,165,159). coords(fra2gt_3,34,200,159).  visible(fra2gt_3,159). happensAt(walking(fra2gt_3),159). orientation(fra2gt_5,150,159). coords(fra2gt_5,29,118,159).  visible(fra2gt_5,159). happensAt(walking(fra2gt_5),159).
orientation(fra2gt_3,165,160). coords(fra2gt_3,35,199,160).  visible(fra2gt_3,160). happensAt(walking(fra2gt_3),160). orientation(fra2gt_5,150,160). coords(fra2gt_5,29,117,160).  visible(fra2gt_5,160). happensAt(walking(fra2gt_5),160).
orientation(fra2gt_3,165,161). coords(fra2gt_3,35,199,161).  visible(fra2gt_3,161). happensAt(walking(fra2gt_3),161). orientation(fra2gt_5,150,161). coords(fra2gt_5,29,117,161).  visible(fra2gt_5,161). happensAt(walking(fra2gt_5),161).
orientation(fra2gt_3,165,162). coords(fra2gt_3,35,198,162).  visible(fra2gt_3,162). happensAt(walking(fra2gt_3),162). orientation(fra2gt_5,150,162). coords(fra2gt_5,29,117,162).  visible(fra2gt_5,162). happensAt(walking(fra2gt_5),162).
orientation(fra2gt_3,165,163). coords(fra2gt_3,36,198,163).  visible(fra2gt_3,163). happensAt(walking(fra2gt_3),163). orientation(fra2gt_5,150,163). coords(fra2gt_5,29,117,163).  visible(fra2gt_5,163). happensAt(walking(fra2gt_5),163).
orientation(fra2gt_3,165,164). coords(fra2gt_3,36,197,164).  visible(fra2gt_3,164). happensAt(walking(fra2gt_3),164). orientation(fra2gt_5,150,164). coords(fra2gt_5,29,117,164).  visible(fra2gt_5,164). happensAt(walking(fra2gt_5),164).
orientation(fra2gt_3,165,165). coords(fra2gt_3,36,197,165).  visible(fra2gt_3,165). happensAt(walking(fra2gt_3),165). orientation(fra2gt_5,150,165). coords(fra2gt_5,29,117,165).  visible(fra2gt_5,165). happensAt(walking(fra2gt_5),165).
orientation(fra2gt_3,165,166). coords(fra2gt_3,36,197,166).  visible(fra2gt_3,166). happensAt(walking(fra2gt_3),166). orientation(fra2gt_5,150,166). coords(fra2gt_5,29,117,166).  visible(fra2gt_5,166). happensAt(walking(fra2gt_5),166).
orientation(fra2gt_3,161,167). coords(fra2gt_3,36,197,167).  visible(fra2gt_3,167). happensAt(walking(fra2gt_3),167). orientation(fra2gt_5,150,167). coords(fra2gt_5,29,117,167).  visible(fra2gt_5,167). happensAt(walking(fra2gt_5),167).
orientation(fra2gt_3,161,168). coords(fra2gt_3,37,196,168).  visible(fra2gt_3,168). happensAt(walking(fra2gt_3),168). orientation(fra2gt_5,150,168). coords(fra2gt_5,29,117,168).  visible(fra2gt_5,168). happensAt(walking(fra2gt_5),168).
orientation(fra2gt_3,161,169). coords(fra2gt_3,37,196,169).  visible(fra2gt_3,169). happensAt(walking(fra2gt_3),169). orientation(fra2gt_5,150,169). coords(fra2gt_5,29,117,169).  visible(fra2gt_5,169). happensAt(walking(fra2gt_5),169).
orientation(fra2gt_3,161,170). coords(fra2gt_3,37,196,170).  visible(fra2gt_3,170). happensAt(walking(fra2gt_3),170). orientation(fra2gt_5,150,170). coords(fra2gt_5,29,117,170).  visible(fra2gt_5,170). happensAt(walking(fra2gt_5),170).
orientation(fra2gt_3,161,171). coords(fra2gt_3,37,195,171).  visible(fra2gt_3,171). happensAt(walking(fra2gt_3),171). orientation(fra2gt_5,150,171). coords(fra2gt_5,29,117,171).  visible(fra2gt_5,171). happensAt(walking(fra2gt_5),171).
orientation(fra2gt_3,161,172). coords(fra2gt_3,37,194,172).  visible(fra2gt_3,172). happensAt(walking(fra2gt_3),172). orientation(fra2gt_5,150,172). coords(fra2gt_5,29,117,172).  visible(fra2gt_5,172). happensAt(walking(fra2gt_5),172).
orientation(fra2gt_3,161,173). coords(fra2gt_3,37,194,173).  visible(fra2gt_3,173). happensAt(walking(fra2gt_3),173). orientation(fra2gt_5,150,173). coords(fra2gt_5,29,117,173).  visible(fra2gt_5,173). happensAt(walking(fra2gt_5),173).
orientation(fra2gt_3,161,174). coords(fra2gt_3,37,194,174).  visible(fra2gt_3,174). happensAt(walking(fra2gt_3),174). orientation(fra2gt_5,150,174). coords(fra2gt_5,29,117,174).  visible(fra2gt_5,174). happensAt(walking(fra2gt_5),174).
orientation(fra2gt_3,161,175). coords(fra2gt_3,37,193,175).  visible(fra2gt_3,175). happensAt(walking(fra2gt_3),175). orientation(fra2gt_5,150,175). coords(fra2gt_5,29,117,175).  visible(fra2gt_5,175). happensAt(walking(fra2gt_5),175).
#show holdsAt/2.