
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

	orientation(mws1gt_3,154,150). coords(mws1gt_3,183,160,150).  visible(mws1gt_3,150). happensAt(walking(mws1gt_3),150). orientation(mws1gt_4,0,150). coords(mws1gt_4,28,133,150).  visible(mws1gt_4,150). happensAt(walking(mws1gt_4),150).
orientation(mws1gt_3,154,151). coords(mws1gt_3,181,160,151).  visible(mws1gt_3,151). happensAt(walking(mws1gt_3),151). orientation(mws1gt_4,0,151). coords(mws1gt_4,28,132,151).  visible(mws1gt_4,151). happensAt(walking(mws1gt_4),151).
orientation(mws1gt_3,154,152). coords(mws1gt_3,179,158,152).  visible(mws1gt_3,152). happensAt(walking(mws1gt_3),152). orientation(mws1gt_4,0,152). coords(mws1gt_4,28,132,152).  visible(mws1gt_4,152). happensAt(walking(mws1gt_4),152).
orientation(mws1gt_3,154,153). coords(mws1gt_3,178,157,153).  visible(mws1gt_3,153). happensAt(walking(mws1gt_3),153). orientation(mws1gt_4,0,153). coords(mws1gt_4,28,131,153).  visible(mws1gt_4,153). happensAt(walking(mws1gt_4),153).
orientation(mws1gt_3,154,154). coords(mws1gt_3,176,155,154).  visible(mws1gt_3,154). happensAt(walking(mws1gt_3),154). orientation(mws1gt_4,0,154). coords(mws1gt_4,28,129,154).  visible(mws1gt_4,154). happensAt(walking(mws1gt_4),154).
orientation(mws1gt_3,154,155). coords(mws1gt_3,174,154,155).  visible(mws1gt_3,155). happensAt(walking(mws1gt_3),155). orientation(mws1gt_4,0,155). coords(mws1gt_4,28,129,155).  visible(mws1gt_4,155). happensAt(walking(mws1gt_4),155).
orientation(mws1gt_3,154,156). coords(mws1gt_3,173,154,156).  visible(mws1gt_3,156). happensAt(walking(mws1gt_3),156). orientation(mws1gt_4,0,156). coords(mws1gt_4,28,127,156).  visible(mws1gt_4,156). happensAt(walking(mws1gt_4),156).
orientation(mws1gt_3,154,157). coords(mws1gt_3,171,154,157).  visible(mws1gt_3,157). happensAt(walking(mws1gt_3),157). orientation(mws1gt_4,0,157). coords(mws1gt_4,28,125,157).  visible(mws1gt_4,157). happensAt(walking(mws1gt_4),157).
orientation(mws1gt_3,154,158). coords(mws1gt_3,171,154,158).  visible(mws1gt_3,158). happensAt(walking(mws1gt_3),158). orientation(mws1gt_4,0,158). coords(mws1gt_4,28,124,158).  visible(mws1gt_4,158). happensAt(walking(mws1gt_4),158).
orientation(mws1gt_3,154,159). coords(mws1gt_3,171,154,159).  visible(mws1gt_3,159). happensAt(walking(mws1gt_3),159). orientation(mws1gt_4,0,159). coords(mws1gt_4,28,124,159).  visible(mws1gt_4,159). happensAt(walking(mws1gt_4),159).
orientation(mws1gt_3,154,160). coords(mws1gt_3,170,154,160).  visible(mws1gt_3,160). happensAt(walking(mws1gt_3),160). orientation(mws1gt_4,0,160). coords(mws1gt_4,28,124,160).  visible(mws1gt_4,160). happensAt(walking(mws1gt_4),160).
orientation(mws1gt_3,154,161). coords(mws1gt_3,170,154,161).  visible(mws1gt_3,161). happensAt(walking(mws1gt_3),161). orientation(mws1gt_4,0,161). coords(mws1gt_4,28,124,161).  visible(mws1gt_4,161). happensAt(walking(mws1gt_4),161).
orientation(mws1gt_3,154,162). coords(mws1gt_3,169,154,162).  visible(mws1gt_3,162). happensAt(walking(mws1gt_3),162). orientation(mws1gt_4,0,162). coords(mws1gt_4,28,124,162).  visible(mws1gt_4,162). happensAt(walking(mws1gt_4),162).
orientation(mws1gt_3,154,163). coords(mws1gt_3,169,154,163).  visible(mws1gt_3,163). happensAt(walking(mws1gt_3),163). orientation(mws1gt_4,0,163). coords(mws1gt_4,28,124,163).  visible(mws1gt_4,163). happensAt(walking(mws1gt_4),163).
orientation(mws1gt_3,154,164). coords(mws1gt_3,169,154,164).  visible(mws1gt_3,164). happensAt(walking(mws1gt_3),164). orientation(mws1gt_4,0,164). coords(mws1gt_4,28,124,164).  visible(mws1gt_4,164). happensAt(walking(mws1gt_4),164).
orientation(mws1gt_3,154,165). coords(mws1gt_3,168,154,165).  visible(mws1gt_3,165). happensAt(walking(mws1gt_3),165). orientation(mws1gt_4,0,165). coords(mws1gt_4,29,123,165).  visible(mws1gt_4,165). happensAt(walking(mws1gt_4),165).
orientation(mws1gt_3,154,166). coords(mws1gt_3,167,154,166).  visible(mws1gt_3,166). happensAt(walking(mws1gt_3),166). orientation(mws1gt_4,0,166). coords(mws1gt_4,29,123,166).  visible(mws1gt_4,166). happensAt(walking(mws1gt_4),166).
orientation(mws1gt_3,154,167). coords(mws1gt_3,166,154,167).  visible(mws1gt_3,167). happensAt(walking(mws1gt_3),167). orientation(mws1gt_4,0,167). coords(mws1gt_4,29,122,167).  visible(mws1gt_4,167). happensAt(walking(mws1gt_4),167).
orientation(mws1gt_3,154,168). coords(mws1gt_3,164,154,168).  visible(mws1gt_3,168). happensAt(walking(mws1gt_3),168). orientation(mws1gt_4,0,168). coords(mws1gt_4,29,122,168).  visible(mws1gt_4,168). happensAt(walking(mws1gt_4),168).
orientation(mws1gt_3,154,169). coords(mws1gt_3,163,155,169).  visible(mws1gt_3,169). happensAt(walking(mws1gt_3),169). orientation(mws1gt_4,0,169). coords(mws1gt_4,29,122,169).  visible(mws1gt_4,169). happensAt(walking(mws1gt_4),169).
orientation(mws1gt_3,154,170). coords(mws1gt_3,162,155,170).  visible(mws1gt_3,170). happensAt(walking(mws1gt_3),170). orientation(mws1gt_4,0,170). coords(mws1gt_4,29,122,170).  visible(mws1gt_4,170). happensAt(walking(mws1gt_4),170).
orientation(mws1gt_3,154,171). coords(mws1gt_3,160,154,171).  visible(mws1gt_3,171). happensAt(walking(mws1gt_3),171). orientation(mws1gt_4,0,171). coords(mws1gt_4,29,121,171).  visible(mws1gt_4,171). happensAt(walking(mws1gt_4),171).
orientation(mws1gt_3,154,172). coords(mws1gt_3,159,154,172).  visible(mws1gt_3,172). happensAt(walking(mws1gt_3),172). orientation(mws1gt_4,0,172). coords(mws1gt_4,29,121,172).  visible(mws1gt_4,172). happensAt(walking(mws1gt_4),172).
orientation(mws1gt_3,154,173). coords(mws1gt_3,159,154,173).  visible(mws1gt_3,173). happensAt(walking(mws1gt_3),173). orientation(mws1gt_4,0,173). coords(mws1gt_4,29,121,173).  visible(mws1gt_4,173). happensAt(walking(mws1gt_4),173).
orientation(mws1gt_3,154,174). coords(mws1gt_3,158,154,174).  visible(mws1gt_3,174). happensAt(walking(mws1gt_3),174). orientation(mws1gt_4,0,174). coords(mws1gt_4,29,121,174).  visible(mws1gt_4,174). happensAt(walking(mws1gt_4),174).
#show holdsAt/2.