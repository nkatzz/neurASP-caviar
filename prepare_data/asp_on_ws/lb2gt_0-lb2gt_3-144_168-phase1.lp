
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

	orientation(lb2gt_0,141,144). coords(lb2gt_0,185,117,144).  visible(lb2gt_0,144). happensAt(active(lb2gt_0),144). orientation(lb2gt_3,159,144). coords(lb2gt_3,6,61,144).  visible(lb2gt_3,144). happensAt(walking(lb2gt_3),144).
orientation(lb2gt_0,141,145). coords(lb2gt_0,184,117,145).  visible(lb2gt_0,145). happensAt(active(lb2gt_0),145). orientation(lb2gt_3,163,145). coords(lb2gt_3,6,62,145).  visible(lb2gt_3,145). happensAt(walking(lb2gt_3),145).
orientation(lb2gt_0,141,146). coords(lb2gt_0,184,117,146).  visible(lb2gt_0,146). happensAt(active(lb2gt_0),146). orientation(lb2gt_3,168,146). coords(lb2gt_3,6,63,146).  visible(lb2gt_3,146). happensAt(walking(lb2gt_3),146).
orientation(lb2gt_0,141,147). coords(lb2gt_0,184,117,147).  visible(lb2gt_0,147). happensAt(active(lb2gt_0),147). orientation(lb2gt_3,168,147). coords(lb2gt_3,6,63,147).  visible(lb2gt_3,147). happensAt(walking(lb2gt_3),147).
orientation(lb2gt_0,141,148). coords(lb2gt_0,184,117,148).  visible(lb2gt_0,148). happensAt(active(lb2gt_0),148). orientation(lb2gt_3,168,148). coords(lb2gt_3,6,63,148).  visible(lb2gt_3,148). happensAt(walking(lb2gt_3),148).
orientation(lb2gt_0,141,149). coords(lb2gt_0,184,117,149).  visible(lb2gt_0,149). happensAt(active(lb2gt_0),149). orientation(lb2gt_3,172,149). coords(lb2gt_3,6,65,149).  visible(lb2gt_3,149). happensAt(walking(lb2gt_3),149).
orientation(lb2gt_0,141,150). coords(lb2gt_0,184,116,150).  visible(lb2gt_0,150). happensAt(active(lb2gt_0),150). orientation(lb2gt_3,170,150). coords(lb2gt_3,6,65,150).  visible(lb2gt_3,150). happensAt(walking(lb2gt_3),150).
orientation(lb2gt_0,141,151). coords(lb2gt_0,184,116,151).  visible(lb2gt_0,151). happensAt(active(lb2gt_0),151). orientation(lb2gt_3,172,151). coords(lb2gt_3,6,65,151).  visible(lb2gt_3,151). happensAt(walking(lb2gt_3),151).
orientation(lb2gt_0,141,152). coords(lb2gt_0,184,116,152).  visible(lb2gt_0,152). happensAt(active(lb2gt_0),152). orientation(lb2gt_3,173,152). coords(lb2gt_3,6,66,152).  visible(lb2gt_3,152). happensAt(walking(lb2gt_3),152).
orientation(lb2gt_0,141,153). coords(lb2gt_0,184,116,153).  visible(lb2gt_0,153). happensAt(active(lb2gt_0),153). orientation(lb2gt_3,175,153). coords(lb2gt_3,6,66,153).  visible(lb2gt_3,153). happensAt(walking(lb2gt_3),153).
orientation(lb2gt_0,141,154). coords(lb2gt_0,185,116,154).  visible(lb2gt_0,154). happensAt(active(lb2gt_0),154). orientation(lb2gt_3,172,154). coords(lb2gt_3,6,66,154).  visible(lb2gt_3,154). happensAt(walking(lb2gt_3),154).
orientation(lb2gt_0,141,155). coords(lb2gt_0,185,116,155).  visible(lb2gt_0,155). happensAt(active(lb2gt_0),155). orientation(lb2gt_3,173,155). coords(lb2gt_3,6,66,155).  visible(lb2gt_3,155). happensAt(walking(lb2gt_3),155).
orientation(lb2gt_0,141,156). coords(lb2gt_0,185,116,156).  visible(lb2gt_0,156). happensAt(active(lb2gt_0),156). orientation(lb2gt_3,171,156). coords(lb2gt_3,5,67,156).  visible(lb2gt_3,156). happensAt(walking(lb2gt_3),156).
orientation(lb2gt_0,141,157). coords(lb2gt_0,185,116,157).  visible(lb2gt_0,157). happensAt(active(lb2gt_0),157). orientation(lb2gt_3,169,157). coords(lb2gt_3,5,67,157).  visible(lb2gt_3,157). happensAt(walking(lb2gt_3),157).
orientation(lb2gt_0,141,158). coords(lb2gt_0,185,116,158).  visible(lb2gt_0,158). happensAt(active(lb2gt_0),158). orientation(lb2gt_3,170,158). coords(lb2gt_3,5,67,158).  visible(lb2gt_3,158). happensAt(walking(lb2gt_3),158).
orientation(lb2gt_0,141,159). coords(lb2gt_0,185,116,159).  visible(lb2gt_0,159). happensAt(active(lb2gt_0),159). orientation(lb2gt_3,172,159). coords(lb2gt_3,5,67,159).  visible(lb2gt_3,159). happensAt(walking(lb2gt_3),159).
orientation(lb2gt_0,141,160). coords(lb2gt_0,185,116,160).  visible(lb2gt_0,160). happensAt(active(lb2gt_0),160). orientation(lb2gt_3,169,160). coords(lb2gt_3,5,67,160).  visible(lb2gt_3,160). happensAt(walking(lb2gt_3),160).
orientation(lb2gt_0,141,161). coords(lb2gt_0,185,116,161).  visible(lb2gt_0,161). happensAt(active(lb2gt_0),161). orientation(lb2gt_3,172,161). coords(lb2gt_3,5,67,161).  visible(lb2gt_3,161). happensAt(walking(lb2gt_3),161).
orientation(lb2gt_0,141,162). coords(lb2gt_0,185,116,162).  visible(lb2gt_0,162). happensAt(active(lb2gt_0),162). orientation(lb2gt_3,174,162). coords(lb2gt_3,5,68,162).  visible(lb2gt_3,162). happensAt(walking(lb2gt_3),162).
orientation(lb2gt_0,141,163). coords(lb2gt_0,185,116,163).  visible(lb2gt_0,163). happensAt(active(lb2gt_0),163). orientation(lb2gt_3,174,163). coords(lb2gt_3,5,68,163).  visible(lb2gt_3,163). happensAt(walking(lb2gt_3),163).
orientation(lb2gt_0,141,164). coords(lb2gt_0,186,116,164).  visible(lb2gt_0,164). happensAt(active(lb2gt_0),164). orientation(lb2gt_3,176,164). coords(lb2gt_3,5,68,164).  visible(lb2gt_3,164). happensAt(walking(lb2gt_3),164).
orientation(lb2gt_0,141,165). coords(lb2gt_0,186,116,165).  visible(lb2gt_0,165). happensAt(active(lb2gt_0),165). orientation(lb2gt_3,174,165). coords(lb2gt_3,5,68,165).  visible(lb2gt_3,165). happensAt(walking(lb2gt_3),165).
orientation(lb2gt_0,141,166). coords(lb2gt_0,186,116,166).  visible(lb2gt_0,166). happensAt(active(lb2gt_0),166). orientation(lb2gt_3,177,166). coords(lb2gt_3,5,68,166).  visible(lb2gt_3,166). happensAt(walking(lb2gt_3),166).
orientation(lb2gt_0,141,167). coords(lb2gt_0,186,116,167).  visible(lb2gt_0,167). happensAt(active(lb2gt_0),167). orientation(lb2gt_3,174,167). coords(lb2gt_3,5,68,167).  visible(lb2gt_3,167). happensAt(walking(lb2gt_3),167).
orientation(lb2gt_0,141,168). coords(lb2gt_0,187,117,168).  visible(lb2gt_0,168). happensAt(active(lb2gt_0),168). orientation(lb2gt_3,0,168). coords(lb2gt_3,5,68,168).  visible(lb2gt_3,168). happensAt(walking(lb2gt_3),168).
#show holdsAt/2.