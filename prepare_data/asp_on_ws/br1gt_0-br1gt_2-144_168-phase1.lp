
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

	orientation(br1gt_0,159,144). coords(br1gt_0,235,189,144).  visible(br1gt_0,144). happensAt(walking(br1gt_0),144). orientation(br1gt_2,171,144). coords(br1gt_2,86,207,144).  visible(br1gt_2,144). happensAt(inactive(br1gt_2),144).
orientation(br1gt_0,159,145). coords(br1gt_0,238,192,145).  visible(br1gt_0,145). happensAt(walking(br1gt_0),145). orientation(br1gt_2,171,145). coords(br1gt_2,86,207,145).  visible(br1gt_2,145). happensAt(inactive(br1gt_2),145).
orientation(br1gt_0,159,146). coords(br1gt_0,241,194,146).  visible(br1gt_0,146). happensAt(walking(br1gt_0),146). orientation(br1gt_2,171,146). coords(br1gt_2,86,207,146).  visible(br1gt_2,146). happensAt(inactive(br1gt_2),146).
orientation(br1gt_0,159,147). coords(br1gt_0,242,197,147).  visible(br1gt_0,147). happensAt(walking(br1gt_0),147). orientation(br1gt_2,171,147). coords(br1gt_2,86,207,147).  visible(br1gt_2,147). happensAt(inactive(br1gt_2),147).
orientation(br1gt_0,161,148). coords(br1gt_0,244,198,148).  visible(br1gt_0,148). happensAt(walking(br1gt_0),148). orientation(br1gt_2,171,148). coords(br1gt_2,86,207,148).  visible(br1gt_2,148). happensAt(inactive(br1gt_2),148).
orientation(br1gt_0,155,149). coords(br1gt_0,244,200,149).  visible(br1gt_0,149). happensAt(walking(br1gt_0),149). orientation(br1gt_2,171,149). coords(br1gt_2,86,207,149).  visible(br1gt_2,149). happensAt(inactive(br1gt_2),149).
orientation(br1gt_0,160,150). coords(br1gt_0,245,201,150).  visible(br1gt_0,150). happensAt(walking(br1gt_0),150). orientation(br1gt_2,171,150). coords(br1gt_2,86,207,150).  visible(br1gt_2,150). happensAt(inactive(br1gt_2),150).
orientation(br1gt_0,161,151). coords(br1gt_0,246,204,151).  visible(br1gt_0,151). happensAt(walking(br1gt_0),151). orientation(br1gt_2,171,151). coords(br1gt_2,86,207,151).  visible(br1gt_2,151). happensAt(inactive(br1gt_2),151).
orientation(br1gt_0,167,152). coords(br1gt_0,247,206,152).  visible(br1gt_0,152). happensAt(walking(br1gt_0),152). orientation(br1gt_2,171,152). coords(br1gt_2,86,207,152).  visible(br1gt_2,152). happensAt(inactive(br1gt_2),152).
orientation(br1gt_0,160,153). coords(br1gt_0,248,208,153).  visible(br1gt_0,153). happensAt(walking(br1gt_0),153). orientation(br1gt_2,171,153). coords(br1gt_2,86,207,153).  visible(br1gt_2,153). happensAt(inactive(br1gt_2),153).
orientation(br1gt_0,160,154). coords(br1gt_0,249,210,154).  visible(br1gt_0,154). happensAt(walking(br1gt_0),154). orientation(br1gt_2,171,154). coords(br1gt_2,86,207,154).  visible(br1gt_2,154). happensAt(inactive(br1gt_2),154).
orientation(br1gt_0,160,155). coords(br1gt_0,251,213,155).  visible(br1gt_0,155). happensAt(walking(br1gt_0),155). orientation(br1gt_2,171,155). coords(br1gt_2,86,207,155).  visible(br1gt_2,155). happensAt(inactive(br1gt_2),155).
orientation(br1gt_0,160,156). coords(br1gt_0,252,216,156).  visible(br1gt_0,156). happensAt(walking(br1gt_0),156). orientation(br1gt_2,171,156). coords(br1gt_2,86,207,156).  visible(br1gt_2,156). happensAt(inactive(br1gt_2),156).
orientation(br1gt_0,156,157). coords(br1gt_0,254,220,157).  visible(br1gt_0,157). happensAt(walking(br1gt_0),157). orientation(br1gt_2,171,157). coords(br1gt_2,86,207,157).  visible(br1gt_2,157). happensAt(inactive(br1gt_2),157).
orientation(br1gt_0,156,158). coords(br1gt_0,257,224,158).  visible(br1gt_0,158). happensAt(walking(br1gt_0),158). orientation(br1gt_2,171,158). coords(br1gt_2,86,207,158).  visible(br1gt_2,158). happensAt(inactive(br1gt_2),158).
orientation(br1gt_0,156,159). coords(br1gt_0,259,226,159).  visible(br1gt_0,159). happensAt(walking(br1gt_0),159). orientation(br1gt_2,171,159). coords(br1gt_2,86,207,159).  visible(br1gt_2,159). happensAt(inactive(br1gt_2),159).
orientation(br1gt_0,156,160). coords(br1gt_0,262,227,160).  visible(br1gt_0,160). happensAt(walking(br1gt_0),160). orientation(br1gt_2,171,160). coords(br1gt_2,86,207,160).  visible(br1gt_2,160). happensAt(inactive(br1gt_2),160).
orientation(br1gt_0,156,161). coords(br1gt_0,264,227,161).  visible(br1gt_0,161). happensAt(walking(br1gt_0),161). orientation(br1gt_2,171,161). coords(br1gt_2,86,207,161).  visible(br1gt_2,161). happensAt(inactive(br1gt_2),161).
orientation(br1gt_0,52,162). coords(br1gt_0,266,227,162).  visible(br1gt_0,162). happensAt(walking(br1gt_0),162). orientation(br1gt_2,171,162). coords(br1gt_2,86,207,162).  visible(br1gt_2,162). happensAt(inactive(br1gt_2),162).
orientation(br1gt_0,52,163). coords(br1gt_0,266,228,163).  visible(br1gt_0,163). happensAt(walking(br1gt_0),163). orientation(br1gt_2,171,163). coords(br1gt_2,86,207,163).  visible(br1gt_2,163). happensAt(inactive(br1gt_2),163).
orientation(br1gt_0,52,164). coords(br1gt_0,268,230,164).  visible(br1gt_0,164). happensAt(walking(br1gt_0),164). orientation(br1gt_2,171,164). coords(br1gt_2,86,207,164).  visible(br1gt_2,164). happensAt(inactive(br1gt_2),164).
orientation(br1gt_0,52,165). coords(br1gt_0,269,233,165).  visible(br1gt_0,165). happensAt(walking(br1gt_0),165). orientation(br1gt_2,171,165). coords(br1gt_2,86,207,165).  visible(br1gt_2,165). happensAt(inactive(br1gt_2),165).
orientation(br1gt_0,47,166). coords(br1gt_0,269,233,166).  visible(br1gt_0,166). happensAt(walking(br1gt_0),166). orientation(br1gt_2,171,166). coords(br1gt_2,86,207,166).  visible(br1gt_2,166). happensAt(inactive(br1gt_2),166).
orientation(br1gt_0,44,167). coords(br1gt_0,271,240,167).  visible(br1gt_0,167). happensAt(walking(br1gt_0),167). orientation(br1gt_2,171,167). coords(br1gt_2,86,207,167).  visible(br1gt_2,167). happensAt(inactive(br1gt_2),167).
orientation(br1gt_0,44,168). coords(br1gt_0,273,243,168).  visible(br1gt_0,168). happensAt(walking(br1gt_0),168). orientation(br1gt_2,171,168). coords(br1gt_2,86,207,168).  visible(br1gt_2,168). happensAt(inactive(br1gt_2),168).
#show holdsAt/2.