
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

	orientation(mws1gt_4,0,142). coords(mws1gt_4,28,134,142).  visible(mws1gt_4,142). happensAt(walking(mws1gt_4),142). orientation(mws1gt_5,145,142). coords(mws1gt_5,75,71,142).  visible(mws1gt_5,142). happensAt(walking(mws1gt_5),142).
orientation(mws1gt_4,0,143). coords(mws1gt_4,28,134,143).  visible(mws1gt_4,143). happensAt(walking(mws1gt_4),143). orientation(mws1gt_5,145,143). coords(mws1gt_5,75,71,143).  visible(mws1gt_5,143). happensAt(walking(mws1gt_5),143).
orientation(mws1gt_4,0,144). coords(mws1gt_4,28,134,144).  visible(mws1gt_4,144). happensAt(walking(mws1gt_4),144). orientation(mws1gt_5,145,144). coords(mws1gt_5,74,71,144).  visible(mws1gt_5,144). happensAt(walking(mws1gt_5),144).
orientation(mws1gt_4,0,145). coords(mws1gt_4,28,134,145).  visible(mws1gt_4,145). happensAt(walking(mws1gt_4),145). orientation(mws1gt_5,145,145). coords(mws1gt_5,74,72,145).  visible(mws1gt_5,145). happensAt(walking(mws1gt_5),145).
orientation(mws1gt_4,0,146). coords(mws1gt_4,28,134,146).  visible(mws1gt_4,146). happensAt(walking(mws1gt_4),146). orientation(mws1gt_5,145,146). coords(mws1gt_5,73,72,146).  visible(mws1gt_5,146). happensAt(walking(mws1gt_5),146).
orientation(mws1gt_4,0,147). coords(mws1gt_4,28,134,147).  visible(mws1gt_4,147). happensAt(walking(mws1gt_4),147). orientation(mws1gt_5,145,147). coords(mws1gt_5,72,73,147).  visible(mws1gt_5,147). happensAt(walking(mws1gt_5),147).
orientation(mws1gt_4,0,148). coords(mws1gt_4,28,133,148).  visible(mws1gt_4,148). happensAt(walking(mws1gt_4),148). orientation(mws1gt_5,145,148). coords(mws1gt_5,71,74,148).  visible(mws1gt_5,148). happensAt(walking(mws1gt_5),148).
orientation(mws1gt_4,0,149). coords(mws1gt_4,28,133,149).  visible(mws1gt_4,149). happensAt(walking(mws1gt_4),149). orientation(mws1gt_5,145,149). coords(mws1gt_5,71,75,149).  visible(mws1gt_5,149). happensAt(walking(mws1gt_5),149).
orientation(mws1gt_4,0,150). coords(mws1gt_4,28,133,150).  visible(mws1gt_4,150). happensAt(walking(mws1gt_4),150). orientation(mws1gt_5,145,150). coords(mws1gt_5,71,77,150).  visible(mws1gt_5,150). happensAt(walking(mws1gt_5),150).
orientation(mws1gt_4,0,151). coords(mws1gt_4,28,132,151).  visible(mws1gt_4,151). happensAt(walking(mws1gt_4),151). orientation(mws1gt_5,145,151). coords(mws1gt_5,71,78,151).  visible(mws1gt_5,151). happensAt(walking(mws1gt_5),151).
orientation(mws1gt_4,0,152). coords(mws1gt_4,28,132,152).  visible(mws1gt_4,152). happensAt(walking(mws1gt_4),152). orientation(mws1gt_5,145,152). coords(mws1gt_5,71,79,152).  visible(mws1gt_5,152). happensAt(walking(mws1gt_5),152).
orientation(mws1gt_4,0,153). coords(mws1gt_4,28,131,153).  visible(mws1gt_4,153). happensAt(walking(mws1gt_4),153). orientation(mws1gt_5,145,153). coords(mws1gt_5,71,80,153).  visible(mws1gt_5,153). happensAt(walking(mws1gt_5),153).
orientation(mws1gt_4,0,154). coords(mws1gt_4,28,129,154).  visible(mws1gt_4,154). happensAt(walking(mws1gt_4),154). orientation(mws1gt_5,145,154). coords(mws1gt_5,71,81,154).  visible(mws1gt_5,154). happensAt(walking(mws1gt_5),154).
orientation(mws1gt_4,0,155). coords(mws1gt_4,28,129,155).  visible(mws1gt_4,155). happensAt(walking(mws1gt_4),155). orientation(mws1gt_5,145,155). coords(mws1gt_5,71,82,155).  visible(mws1gt_5,155). happensAt(walking(mws1gt_5),155).
orientation(mws1gt_4,0,156). coords(mws1gt_4,28,127,156).  visible(mws1gt_4,156). happensAt(walking(mws1gt_4),156). orientation(mws1gt_5,145,156). coords(mws1gt_5,71,82,156).  visible(mws1gt_5,156). happensAt(walking(mws1gt_5),156).
orientation(mws1gt_4,0,157). coords(mws1gt_4,28,125,157).  visible(mws1gt_4,157). happensAt(walking(mws1gt_4),157). orientation(mws1gt_5,145,157). coords(mws1gt_5,71,82,157).  visible(mws1gt_5,157). happensAt(walking(mws1gt_5),157).
orientation(mws1gt_4,0,158). coords(mws1gt_4,28,124,158).  visible(mws1gt_4,158). happensAt(walking(mws1gt_4),158). orientation(mws1gt_5,145,158). coords(mws1gt_5,71,83,158).  visible(mws1gt_5,158). happensAt(walking(mws1gt_5),158).
orientation(mws1gt_4,0,159). coords(mws1gt_4,28,124,159).  visible(mws1gt_4,159). happensAt(walking(mws1gt_4),159). orientation(mws1gt_5,145,159). coords(mws1gt_5,71,83,159).  visible(mws1gt_5,159). happensAt(walking(mws1gt_5),159).
orientation(mws1gt_4,0,160). coords(mws1gt_4,28,124,160).  visible(mws1gt_4,160). happensAt(walking(mws1gt_4),160). orientation(mws1gt_5,145,160). coords(mws1gt_5,70,83,160).  visible(mws1gt_5,160). happensAt(walking(mws1gt_5),160).
orientation(mws1gt_4,0,161). coords(mws1gt_4,28,124,161).  visible(mws1gt_4,161). happensAt(walking(mws1gt_4),161). orientation(mws1gt_5,145,161). coords(mws1gt_5,68,83,161).  visible(mws1gt_5,161). happensAt(walking(mws1gt_5),161).
orientation(mws1gt_4,0,162). coords(mws1gt_4,28,124,162).  visible(mws1gt_4,162). happensAt(walking(mws1gt_4),162). orientation(mws1gt_5,145,162). coords(mws1gt_5,67,83,162).  visible(mws1gt_5,162). happensAt(walking(mws1gt_5),162).
orientation(mws1gt_4,0,163). coords(mws1gt_4,28,124,163).  visible(mws1gt_4,163). happensAt(walking(mws1gt_4),163). orientation(mws1gt_5,145,163). coords(mws1gt_5,67,83,163).  visible(mws1gt_5,163). happensAt(walking(mws1gt_5),163).
orientation(mws1gt_4,0,164). coords(mws1gt_4,28,124,164).  visible(mws1gt_4,164). happensAt(walking(mws1gt_4),164). orientation(mws1gt_5,145,164). coords(mws1gt_5,66,84,164).  visible(mws1gt_5,164). happensAt(walking(mws1gt_5),164).
orientation(mws1gt_4,0,165). coords(mws1gt_4,29,123,165).  visible(mws1gt_4,165). happensAt(walking(mws1gt_4),165). orientation(mws1gt_5,145,165). coords(mws1gt_5,65,86,165).  visible(mws1gt_5,165). happensAt(walking(mws1gt_5),165).
orientation(mws1gt_4,0,166). coords(mws1gt_4,29,123,166).  visible(mws1gt_4,166). happensAt(walking(mws1gt_4),166). orientation(mws1gt_5,145,166). coords(mws1gt_5,64,87,166).  visible(mws1gt_5,166). happensAt(walking(mws1gt_5),166).
#show holdsAt/2.