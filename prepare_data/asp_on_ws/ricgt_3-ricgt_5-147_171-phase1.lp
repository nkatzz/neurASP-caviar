
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

	orientation(ricgt_3,150,147). coords(ricgt_3,190,158,147).  visible(ricgt_3,147). happensAt(active(ricgt_3),147). orientation(ricgt_5,0,147). coords(ricgt_5,27,129,147).  visible(ricgt_5,147). happensAt(walking(ricgt_5),147).
orientation(ricgt_3,149,148). coords(ricgt_3,190,158,148).  visible(ricgt_3,148). happensAt(active(ricgt_3),148). orientation(ricgt_5,0,148). coords(ricgt_5,27,129,148).  visible(ricgt_5,148). happensAt(walking(ricgt_5),148).
orientation(ricgt_3,147,149). coords(ricgt_3,191,159,149).  visible(ricgt_3,149). happensAt(active(ricgt_3),149). orientation(ricgt_5,0,149). coords(ricgt_5,27,129,149).  visible(ricgt_5,149). happensAt(walking(ricgt_5),149).
orientation(ricgt_3,144,150). coords(ricgt_3,193,161,150).  visible(ricgt_3,150). happensAt(active(ricgt_3),150). orientation(ricgt_5,0,150). coords(ricgt_5,27,130,150).  visible(ricgt_5,150). happensAt(walking(ricgt_5),150).
orientation(ricgt_3,148,151). coords(ricgt_3,195,163,151).  visible(ricgt_3,151). happensAt(active(ricgt_3),151). orientation(ricgt_5,0,151). coords(ricgt_5,27,130,151).  visible(ricgt_5,151). happensAt(walking(ricgt_5),151).
orientation(ricgt_3,150,152). coords(ricgt_3,199,165,152).  visible(ricgt_3,152). happensAt(active(ricgt_3),152). orientation(ricgt_5,0,152). coords(ricgt_5,27,130,152).  visible(ricgt_5,152). happensAt(walking(ricgt_5),152).
orientation(ricgt_3,152,153). coords(ricgt_3,201,167,153).  visible(ricgt_3,153). happensAt(active(ricgt_3),153). orientation(ricgt_5,0,153). coords(ricgt_5,27,130,153).  visible(ricgt_5,153). happensAt(walking(ricgt_5),153).
orientation(ricgt_3,155,154). coords(ricgt_3,203,169,154).  visible(ricgt_3,154). happensAt(active(ricgt_3),154). orientation(ricgt_5,0,154). coords(ricgt_5,27,131,154).  visible(ricgt_5,154). happensAt(walking(ricgt_5),154).
orientation(ricgt_3,159,155). coords(ricgt_3,204,172,155).  visible(ricgt_3,155). happensAt(active(ricgt_3),155). orientation(ricgt_5,0,155). coords(ricgt_5,27,131,155).  visible(ricgt_5,155). happensAt(walking(ricgt_5),155).
orientation(ricgt_3,155,156). coords(ricgt_3,207,174,156).  visible(ricgt_3,156). happensAt(walking(ricgt_3),156). orientation(ricgt_5,0,156). coords(ricgt_5,27,132,156).  visible(ricgt_5,156). happensAt(walking(ricgt_5),156).
orientation(ricgt_3,154,157). coords(ricgt_3,207,175,157).  visible(ricgt_3,157). happensAt(walking(ricgt_3),157). orientation(ricgt_5,0,157). coords(ricgt_5,27,132,157).  visible(ricgt_5,157). happensAt(walking(ricgt_5),157).
orientation(ricgt_3,154,158). coords(ricgt_3,208,176,158).  visible(ricgt_3,158). happensAt(walking(ricgt_3),158). orientation(ricgt_5,0,158). coords(ricgt_5,27,132,158).  visible(ricgt_5,158). happensAt(walking(ricgt_5),158).
orientation(ricgt_3,159,159). coords(ricgt_3,209,178,159).  visible(ricgt_3,159). happensAt(walking(ricgt_3),159). orientation(ricgt_5,0,159). coords(ricgt_5,27,132,159).  visible(ricgt_5,159). happensAt(walking(ricgt_5),159).
orientation(ricgt_3,162,160). coords(ricgt_3,211,178,160).  visible(ricgt_3,160). happensAt(walking(ricgt_3),160). orientation(ricgt_5,0,160). coords(ricgt_5,27,133,160).  visible(ricgt_5,160). happensAt(walking(ricgt_5),160).
orientation(ricgt_3,166,161). coords(ricgt_3,211,179,161).  visible(ricgt_3,161). happensAt(walking(ricgt_3),161). orientation(ricgt_5,0,161). coords(ricgt_5,27,133,161).  visible(ricgt_5,161). happensAt(walking(ricgt_5),161).
orientation(ricgt_3,168,162). coords(ricgt_3,212,182,162).  visible(ricgt_3,162). happensAt(walking(ricgt_3),162). orientation(ricgt_5,0,162). coords(ricgt_5,27,133,162).  visible(ricgt_5,162). happensAt(walking(ricgt_5),162).
orientation(ricgt_3,169,163). coords(ricgt_3,214,184,163).  visible(ricgt_3,163). happensAt(walking(ricgt_3),163). orientation(ricgt_5,0,163). coords(ricgt_5,27,133,163).  visible(ricgt_5,163). happensAt(walking(ricgt_5),163).
orientation(ricgt_3,170,164). coords(ricgt_3,216,185,164).  visible(ricgt_3,164). happensAt(walking(ricgt_3),164). orientation(ricgt_5,0,164). coords(ricgt_5,27,133,164).  visible(ricgt_5,164). happensAt(walking(ricgt_5),164).
orientation(ricgt_3,172,165). coords(ricgt_3,220,186,165).  visible(ricgt_3,165). happensAt(walking(ricgt_3),165). orientation(ricgt_5,0,165). coords(ricgt_5,27,133,165).  visible(ricgt_5,165). happensAt(walking(ricgt_5),165).
orientation(ricgt_3,174,166). coords(ricgt_3,223,187,166).  visible(ricgt_3,166). happensAt(walking(ricgt_3),166). orientation(ricgt_5,0,166). coords(ricgt_5,27,133,166).  visible(ricgt_5,166). happensAt(walking(ricgt_5),166).
orientation(ricgt_3,178,167). coords(ricgt_3,228,189,167).  visible(ricgt_3,167). happensAt(walking(ricgt_3),167). orientation(ricgt_5,0,167). coords(ricgt_5,27,133,167).  visible(ricgt_5,167). happensAt(walking(ricgt_5),167).
orientation(ricgt_3,179,168). coords(ricgt_3,230,191,168).  visible(ricgt_3,168). happensAt(walking(ricgt_3),168). orientation(ricgt_5,0,168). coords(ricgt_5,26,134,168).  visible(ricgt_5,168). happensAt(walking(ricgt_5),168).
orientation(ricgt_3,0,169). coords(ricgt_3,233,195,169).  visible(ricgt_3,169). happensAt(walking(ricgt_3),169). orientation(ricgt_5,0,169). coords(ricgt_5,26,134,169).  visible(ricgt_5,169). happensAt(walking(ricgt_5),169).
orientation(ricgt_3,0,170). coords(ricgt_3,235,197,170).  visible(ricgt_3,170). happensAt(walking(ricgt_3),170). orientation(ricgt_5,0,170). coords(ricgt_5,26,134,170).  visible(ricgt_5,170). happensAt(walking(ricgt_5),170).
orientation(ricgt_3,3,171). coords(ricgt_3,236,198,171).  visible(ricgt_3,171). happensAt(walking(ricgt_3),171). orientation(ricgt_5,0,171). coords(ricgt_5,26,135,171).  visible(ricgt_5,171). happensAt(walking(ricgt_5),171).
#show holdsAt/2.