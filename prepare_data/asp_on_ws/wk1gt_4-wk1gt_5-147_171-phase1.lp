
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

	orientation(wk1gt_4,0,147). coords(wk1gt_4,46,183,147).  visible(wk1gt_4,147). happensAt(walking(wk1gt_4),147). orientation(wk1gt_5,162,147). coords(wk1gt_5,45,167,147).  visible(wk1gt_5,147). happensAt(walking(wk1gt_5),147).
orientation(wk1gt_4,0,148). coords(wk1gt_4,46,183,148).  visible(wk1gt_4,148). happensAt(walking(wk1gt_4),148). orientation(wk1gt_5,162,148). coords(wk1gt_5,45,167,148).  visible(wk1gt_5,148). happensAt(walking(wk1gt_5),148).
orientation(wk1gt_4,0,149). coords(wk1gt_4,47,183,149).  visible(wk1gt_4,149). happensAt(walking(wk1gt_4),149). orientation(wk1gt_5,162,149). coords(wk1gt_5,45,167,149).  visible(wk1gt_5,149). happensAt(walking(wk1gt_5),149).
orientation(wk1gt_4,163,150). coords(wk1gt_4,47,183,150).  visible(wk1gt_4,150). happensAt(walking(wk1gt_4),150). orientation(wk1gt_5,162,150). coords(wk1gt_5,45,167,150).  visible(wk1gt_5,150). happensAt(walking(wk1gt_5),150).
orientation(wk1gt_4,166,151). coords(wk1gt_4,47,183,151).  visible(wk1gt_4,151). happensAt(walking(wk1gt_4),151). orientation(wk1gt_5,162,151). coords(wk1gt_5,45,167,151).  visible(wk1gt_5,151). happensAt(walking(wk1gt_5),151).
orientation(wk1gt_4,162,152). coords(wk1gt_4,47,183,152).  visible(wk1gt_4,152). happensAt(walking(wk1gt_4),152). orientation(wk1gt_5,162,152). coords(wk1gt_5,45,167,152).  visible(wk1gt_5,152). happensAt(walking(wk1gt_5),152).
orientation(wk1gt_4,158,153). coords(wk1gt_4,47,183,153).  visible(wk1gt_4,153). happensAt(walking(wk1gt_4),153). orientation(wk1gt_5,162,153). coords(wk1gt_5,45,167,153).  visible(wk1gt_5,153). happensAt(walking(wk1gt_5),153).
orientation(wk1gt_4,163,154). coords(wk1gt_4,47,183,154).  visible(wk1gt_4,154). happensAt(walking(wk1gt_4),154). orientation(wk1gt_5,162,154). coords(wk1gt_5,45,167,154).  visible(wk1gt_5,154). happensAt(walking(wk1gt_5),154).
orientation(wk1gt_4,157,155). coords(wk1gt_4,47,183,155).  visible(wk1gt_4,155). happensAt(walking(wk1gt_4),155). orientation(wk1gt_5,162,155). coords(wk1gt_5,45,167,155).  visible(wk1gt_5,155). happensAt(active(wk1gt_5),155).
orientation(wk1gt_4,164,156). coords(wk1gt_4,47,183,156).  visible(wk1gt_4,156). happensAt(walking(wk1gt_4),156). orientation(wk1gt_5,162,156). coords(wk1gt_5,46,166,156).  visible(wk1gt_5,156). happensAt(active(wk1gt_5),156).
orientation(wk1gt_4,164,157). coords(wk1gt_4,47,183,157).  visible(wk1gt_4,157). happensAt(walking(wk1gt_4),157). orientation(wk1gt_5,162,157). coords(wk1gt_5,46,166,157).  visible(wk1gt_5,157). happensAt(active(wk1gt_5),157).
orientation(wk1gt_4,160,158). coords(wk1gt_4,47,183,158).  visible(wk1gt_4,158). happensAt(walking(wk1gt_4),158). orientation(wk1gt_5,162,158). coords(wk1gt_5,46,166,158).  visible(wk1gt_5,158). happensAt(active(wk1gt_5),158).
orientation(wk1gt_4,160,159). coords(wk1gt_4,47,183,159).  visible(wk1gt_4,159). happensAt(walking(wk1gt_4),159). orientation(wk1gt_5,162,159). coords(wk1gt_5,46,166,159).  visible(wk1gt_5,159). happensAt(active(wk1gt_5),159).
orientation(wk1gt_4,163,160). coords(wk1gt_4,47,183,160).  visible(wk1gt_4,160). happensAt(walking(wk1gt_4),160). orientation(wk1gt_5,162,160). coords(wk1gt_5,46,166,160).  visible(wk1gt_5,160). happensAt(active(wk1gt_5),160).
orientation(wk1gt_4,165,161). coords(wk1gt_4,47,183,161).  visible(wk1gt_4,161). happensAt(walking(wk1gt_4),161). orientation(wk1gt_5,162,161). coords(wk1gt_5,47,166,161).  visible(wk1gt_5,161). happensAt(active(wk1gt_5),161).
orientation(wk1gt_4,161,162). coords(wk1gt_4,47,183,162).  visible(wk1gt_4,162). happensAt(walking(wk1gt_4),162). orientation(wk1gt_5,162,162). coords(wk1gt_5,47,166,162).  visible(wk1gt_5,162). happensAt(active(wk1gt_5),162).
orientation(wk1gt_4,162,163). coords(wk1gt_4,47,182,163).  visible(wk1gt_4,163). happensAt(walking(wk1gt_4),163). orientation(wk1gt_5,162,163). coords(wk1gt_5,47,166,163).  visible(wk1gt_5,163). happensAt(active(wk1gt_5),163).
orientation(wk1gt_4,161,164). coords(wk1gt_4,47,182,164).  visible(wk1gt_4,164). happensAt(walking(wk1gt_4),164). orientation(wk1gt_5,162,164). coords(wk1gt_5,47,166,164).  visible(wk1gt_5,164). happensAt(active(wk1gt_5),164).
orientation(wk1gt_4,163,165). coords(wk1gt_4,47,182,165).  visible(wk1gt_4,165). happensAt(walking(wk1gt_4),165). orientation(wk1gt_5,162,165). coords(wk1gt_5,47,166,165).  visible(wk1gt_5,165). happensAt(active(wk1gt_5),165).
orientation(wk1gt_4,163,166). coords(wk1gt_4,47,182,166).  visible(wk1gt_4,166). happensAt(walking(wk1gt_4),166). orientation(wk1gt_5,162,166). coords(wk1gt_5,47,166,166).  visible(wk1gt_5,166). happensAt(active(wk1gt_5),166).
orientation(wk1gt_4,165,167). coords(wk1gt_4,47,182,167).  visible(wk1gt_4,167). happensAt(walking(wk1gt_4),167). orientation(wk1gt_5,162,167). coords(wk1gt_5,47,166,167).  visible(wk1gt_5,167). happensAt(active(wk1gt_5),167).
orientation(wk1gt_4,159,168). coords(wk1gt_4,47,182,168).  visible(wk1gt_4,168). happensAt(walking(wk1gt_4),168). orientation(wk1gt_5,162,168). coords(wk1gt_5,48,166,168).  visible(wk1gt_5,168). happensAt(active(wk1gt_5),168).
orientation(wk1gt_4,161,169). coords(wk1gt_4,47,181,169).  visible(wk1gt_4,169). happensAt(walking(wk1gt_4),169). orientation(wk1gt_5,162,169). coords(wk1gt_5,48,166,169).  visible(wk1gt_5,169). happensAt(active(wk1gt_5),169).
orientation(wk1gt_4,161,170). coords(wk1gt_4,47,181,170).  visible(wk1gt_4,170). happensAt(active(wk1gt_4),170). orientation(wk1gt_5,162,170). coords(wk1gt_5,48,166,170).  visible(wk1gt_5,170). happensAt(active(wk1gt_5),170).
orientation(wk1gt_4,158,171). coords(wk1gt_4,47,181,171).  visible(wk1gt_4,171). happensAt(active(wk1gt_4),171). orientation(wk1gt_5,162,171). coords(wk1gt_5,48,166,171).  visible(wk1gt_5,171). happensAt(active(wk1gt_5),171).
#show holdsAt/2.