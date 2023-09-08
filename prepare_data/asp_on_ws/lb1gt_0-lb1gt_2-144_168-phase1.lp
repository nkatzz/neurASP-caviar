
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

	orientation(lb1gt_0,136,144). coords(lb1gt_0,170,93,144).  visible(lb1gt_0,144). happensAt(walking(lb1gt_0),144). orientation(lb1gt_2,144,144). coords(lb1gt_2,100,102,144).  visible(lb1gt_2,144). happensAt(walking(lb1gt_2),144).
orientation(lb1gt_0,136,145). coords(lb1gt_0,169,92,145).  visible(lb1gt_0,145). happensAt(walking(lb1gt_0),145). orientation(lb1gt_2,144,145). coords(lb1gt_2,100,103,145).  visible(lb1gt_2,145). happensAt(walking(lb1gt_2),145).
orientation(lb1gt_0,136,146). coords(lb1gt_0,169,90,146).  visible(lb1gt_0,146). happensAt(walking(lb1gt_0),146). orientation(lb1gt_2,144,146). coords(lb1gt_2,100,104,146).  visible(lb1gt_2,146). happensAt(walking(lb1gt_2),146).
orientation(lb1gt_0,136,147). coords(lb1gt_0,169,88,147).  visible(lb1gt_0,147). happensAt(walking(lb1gt_0),147). orientation(lb1gt_2,144,147). coords(lb1gt_2,100,104,147).  visible(lb1gt_2,147). happensAt(walking(lb1gt_2),147).
orientation(lb1gt_0,136,148). coords(lb1gt_0,168,85,148).  visible(lb1gt_0,148). happensAt(walking(lb1gt_0),148). orientation(lb1gt_2,144,148). coords(lb1gt_2,101,106,148).  visible(lb1gt_2,148). happensAt(walking(lb1gt_2),148).
orientation(lb1gt_0,136,149). coords(lb1gt_0,168,82,149).  visible(lb1gt_0,149). happensAt(walking(lb1gt_0),149). orientation(lb1gt_2,144,149). coords(lb1gt_2,101,106,149).  visible(lb1gt_2,149). happensAt(walking(lb1gt_2),149).
orientation(lb1gt_0,136,150). coords(lb1gt_0,166,80,150).  visible(lb1gt_0,150). happensAt(walking(lb1gt_0),150). orientation(lb1gt_2,144,150). coords(lb1gt_2,101,107,150).  visible(lb1gt_2,150). happensAt(walking(lb1gt_2),150).
orientation(lb1gt_0,136,151). coords(lb1gt_0,163,79,151).  visible(lb1gt_0,151). happensAt(walking(lb1gt_0),151). orientation(lb1gt_2,144,151). coords(lb1gt_2,102,108,151).  visible(lb1gt_2,151). happensAt(walking(lb1gt_2),151).
orientation(lb1gt_0,136,152). coords(lb1gt_0,162,77,152).  visible(lb1gt_0,152). happensAt(walking(lb1gt_0),152). orientation(lb1gt_2,144,152). coords(lb1gt_2,102,108,152).  visible(lb1gt_2,152). happensAt(walking(lb1gt_2),152).
orientation(lb1gt_0,136,153). coords(lb1gt_0,162,76,153).  visible(lb1gt_0,153). happensAt(walking(lb1gt_0),153). orientation(lb1gt_2,144,153). coords(lb1gt_2,103,109,153).  visible(lb1gt_2,153). happensAt(walking(lb1gt_2),153).
orientation(lb1gt_0,136,154). coords(lb1gt_0,162,73,154).  visible(lb1gt_0,154). happensAt(walking(lb1gt_0),154). orientation(lb1gt_2,144,154). coords(lb1gt_2,103,109,154).  visible(lb1gt_2,154). happensAt(walking(lb1gt_2),154).
orientation(lb1gt_0,136,155). coords(lb1gt_0,162,72,155).  visible(lb1gt_0,155). happensAt(walking(lb1gt_0),155). orientation(lb1gt_2,144,155). coords(lb1gt_2,103,109,155).  visible(lb1gt_2,155). happensAt(walking(lb1gt_2),155).
orientation(lb1gt_0,136,156). coords(lb1gt_0,161,71,156).  visible(lb1gt_0,156). happensAt(walking(lb1gt_0),156). orientation(lb1gt_2,144,156). coords(lb1gt_2,103,109,156).  visible(lb1gt_2,156). happensAt(walking(lb1gt_2),156).
orientation(lb1gt_0,136,157). coords(lb1gt_0,162,69,157).  visible(lb1gt_0,157). happensAt(walking(lb1gt_0),157). orientation(lb1gt_2,144,157). coords(lb1gt_2,103,109,157).  visible(lb1gt_2,157). happensAt(walking(lb1gt_2),157).
orientation(lb1gt_0,136,158). coords(lb1gt_0,161,67,158).  visible(lb1gt_0,158). happensAt(walking(lb1gt_0),158). orientation(lb1gt_2,144,158). coords(lb1gt_2,103,110,158).  visible(lb1gt_2,158). happensAt(walking(lb1gt_2),158).
orientation(lb1gt_0,136,159). coords(lb1gt_0,162,65,159).  visible(lb1gt_0,159). happensAt(walking(lb1gt_0),159). orientation(lb1gt_2,144,159). coords(lb1gt_2,104,110,159).  visible(lb1gt_2,159). happensAt(walking(lb1gt_2),159).
orientation(lb1gt_0,136,160). coords(lb1gt_0,161,64,160).  visible(lb1gt_0,160). happensAt(walking(lb1gt_0),160). orientation(lb1gt_2,144,160). coords(lb1gt_2,104,110,160).  visible(lb1gt_2,160). happensAt(walking(lb1gt_2),160).
orientation(lb1gt_0,136,161). coords(lb1gt_0,161,63,161).  visible(lb1gt_0,161). happensAt(walking(lb1gt_0),161). orientation(lb1gt_2,144,161). coords(lb1gt_2,104,110,161).  visible(lb1gt_2,161). happensAt(walking(lb1gt_2),161).
orientation(lb1gt_0,136,162). coords(lb1gt_0,160,62,162).  visible(lb1gt_0,162). happensAt(walking(lb1gt_0),162). orientation(lb1gt_2,144,162). coords(lb1gt_2,105,110,162).  visible(lb1gt_2,162). happensAt(walking(lb1gt_2),162).
orientation(lb1gt_0,136,163). coords(lb1gt_0,160,61,163).  visible(lb1gt_0,163). happensAt(walking(lb1gt_0),163). orientation(lb1gt_2,144,163). coords(lb1gt_2,106,110,163).  visible(lb1gt_2,163). happensAt(walking(lb1gt_2),163).
orientation(lb1gt_0,136,164). coords(lb1gt_0,160,59,164).  visible(lb1gt_0,164). happensAt(walking(lb1gt_0),164). orientation(lb1gt_2,144,164). coords(lb1gt_2,108,110,164).  visible(lb1gt_2,164). happensAt(walking(lb1gt_2),164).
orientation(lb1gt_0,136,165). coords(lb1gt_0,161,59,165).  visible(lb1gt_0,165). happensAt(walking(lb1gt_0),165). orientation(lb1gt_2,144,165). coords(lb1gt_2,109,110,165).  visible(lb1gt_2,165). happensAt(walking(lb1gt_2),165).
orientation(lb1gt_0,136,166). coords(lb1gt_0,161,57,166).  visible(lb1gt_0,166). happensAt(walking(lb1gt_0),166). orientation(lb1gt_2,144,166). coords(lb1gt_2,110,111,166).  visible(lb1gt_2,166). happensAt(walking(lb1gt_2),166).
orientation(lb1gt_0,136,167). coords(lb1gt_0,161,55,167).  visible(lb1gt_0,167). happensAt(walking(lb1gt_0),167). orientation(lb1gt_2,144,167). coords(lb1gt_2,111,111,167).  visible(lb1gt_2,167). happensAt(walking(lb1gt_2),167).
orientation(lb1gt_0,136,168). coords(lb1gt_0,161,53,168).  visible(lb1gt_0,168). happensAt(walking(lb1gt_0),168). orientation(lb1gt_2,144,168). coords(lb1gt_2,111,111,168).  visible(lb1gt_2,168). happensAt(walking(lb1gt_2),168).
#show holdsAt/2.