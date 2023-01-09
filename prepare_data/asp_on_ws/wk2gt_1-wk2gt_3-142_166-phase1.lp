
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

	orientation(wk2gt_1,132,142). coords(wk2gt_1,114,53,142).  visible(wk2gt_1,142). happensAt(inactive(wk2gt_1),142). orientation(wk2gt_3,154,142). coords(wk2gt_3,179,158,142).  visible(wk2gt_3,142). happensAt(active(wk2gt_3),142).
orientation(wk2gt_1,132,143). coords(wk2gt_1,114,53,143).  visible(wk2gt_1,143). happensAt(inactive(wk2gt_1),143). orientation(wk2gt_3,154,143). coords(wk2gt_3,178,158,143).  visible(wk2gt_3,143). happensAt(inactive(wk2gt_3),143).
orientation(wk2gt_1,132,144). coords(wk2gt_1,114,53,144).  visible(wk2gt_1,144). happensAt(inactive(wk2gt_1),144). orientation(wk2gt_3,154,144). coords(wk2gt_3,178,158,144).  visible(wk2gt_3,144). happensAt(inactive(wk2gt_3),144).
orientation(wk2gt_1,132,145). coords(wk2gt_1,114,53,145).  visible(wk2gt_1,145). happensAt(inactive(wk2gt_1),145). orientation(wk2gt_3,154,145). coords(wk2gt_3,178,158,145).  visible(wk2gt_3,145). happensAt(inactive(wk2gt_3),145).
orientation(wk2gt_1,132,146). coords(wk2gt_1,114,53,146).  visible(wk2gt_1,146). happensAt(inactive(wk2gt_1),146). orientation(wk2gt_3,154,146). coords(wk2gt_3,178,157,146).  visible(wk2gt_3,146). happensAt(inactive(wk2gt_3),146).
orientation(wk2gt_1,132,147). coords(wk2gt_1,114,53,147).  visible(wk2gt_1,147). happensAt(inactive(wk2gt_1),147). orientation(wk2gt_3,154,147). coords(wk2gt_3,178,157,147).  visible(wk2gt_3,147). happensAt(inactive(wk2gt_3),147).
orientation(wk2gt_1,132,148). coords(wk2gt_1,114,53,148).  visible(wk2gt_1,148). happensAt(inactive(wk2gt_1),148). orientation(wk2gt_3,154,148). coords(wk2gt_3,178,157,148).  visible(wk2gt_3,148). happensAt(inactive(wk2gt_3),148).
orientation(wk2gt_1,132,149). coords(wk2gt_1,114,53,149).  visible(wk2gt_1,149). happensAt(inactive(wk2gt_1),149). orientation(wk2gt_3,154,149). coords(wk2gt_3,178,156,149).  visible(wk2gt_3,149). happensAt(inactive(wk2gt_3),149).
orientation(wk2gt_1,132,150). coords(wk2gt_1,114,53,150).  visible(wk2gt_1,150). happensAt(inactive(wk2gt_1),150). orientation(wk2gt_3,154,150). coords(wk2gt_3,178,156,150).  visible(wk2gt_3,150). happensAt(inactive(wk2gt_3),150).
orientation(wk2gt_1,132,151). coords(wk2gt_1,114,53,151).  visible(wk2gt_1,151). happensAt(inactive(wk2gt_1),151). orientation(wk2gt_3,154,151). coords(wk2gt_3,178,156,151).  visible(wk2gt_3,151). happensAt(inactive(wk2gt_3),151).
orientation(wk2gt_1,132,152). coords(wk2gt_1,114,53,152).  visible(wk2gt_1,152). happensAt(inactive(wk2gt_1),152). orientation(wk2gt_3,154,152). coords(wk2gt_3,178,156,152).  visible(wk2gt_3,152). happensAt(inactive(wk2gt_3),152).
orientation(wk2gt_1,132,153). coords(wk2gt_1,114,53,153).  visible(wk2gt_1,153). happensAt(inactive(wk2gt_1),153). orientation(wk2gt_3,154,153). coords(wk2gt_3,178,156,153).  visible(wk2gt_3,153). happensAt(inactive(wk2gt_3),153).
orientation(wk2gt_1,129,154). coords(wk2gt_1,114,53,154).  visible(wk2gt_1,154). happensAt(inactive(wk2gt_1),154). orientation(wk2gt_3,154,154). coords(wk2gt_3,180,156,154).  visible(wk2gt_3,154). happensAt(inactive(wk2gt_3),154).
orientation(wk2gt_1,129,155). coords(wk2gt_1,114,53,155).  visible(wk2gt_1,155). happensAt(inactive(wk2gt_1),155). orientation(wk2gt_3,149,155). coords(wk2gt_3,180,156,155).  visible(wk2gt_3,155). happensAt(inactive(wk2gt_3),155).
orientation(wk2gt_1,129,156). coords(wk2gt_1,114,53,156).  visible(wk2gt_1,156). happensAt(inactive(wk2gt_1),156). orientation(wk2gt_3,145,156). coords(wk2gt_3,182,156,156).  visible(wk2gt_3,156). happensAt(active(wk2gt_3),156).
orientation(wk2gt_1,129,157). coords(wk2gt_1,114,53,157).  visible(wk2gt_1,157). happensAt(inactive(wk2gt_1),157). orientation(wk2gt_3,145,157). coords(wk2gt_3,183,156,157).  visible(wk2gt_3,157). happensAt(active(wk2gt_3),157).
orientation(wk2gt_1,129,158). coords(wk2gt_1,114,53,158).  visible(wk2gt_1,158). happensAt(inactive(wk2gt_1),158). orientation(wk2gt_3,145,158). coords(wk2gt_3,185,157,158).  visible(wk2gt_3,158). happensAt(active(wk2gt_3),158).
orientation(wk2gt_1,129,159). coords(wk2gt_1,115,53,159).  visible(wk2gt_1,159). happensAt(inactive(wk2gt_1),159). orientation(wk2gt_3,145,159). coords(wk2gt_3,185,157,159).  visible(wk2gt_3,159). happensAt(active(wk2gt_3),159).
orientation(wk2gt_1,129,160). coords(wk2gt_1,115,53,160).  visible(wk2gt_1,160). happensAt(inactive(wk2gt_1),160). orientation(wk2gt_3,145,160). coords(wk2gt_3,186,158,160).  visible(wk2gt_3,160). happensAt(active(wk2gt_3),160).
orientation(wk2gt_1,129,161). coords(wk2gt_1,115,53,161).  visible(wk2gt_1,161). happensAt(inactive(wk2gt_1),161). orientation(wk2gt_3,145,161). coords(wk2gt_3,187,159,161).  visible(wk2gt_3,161). happensAt(active(wk2gt_3),161).
orientation(wk2gt_1,129,162). coords(wk2gt_1,115,53,162).  visible(wk2gt_1,162). happensAt(inactive(wk2gt_1),162). orientation(wk2gt_3,145,162). coords(wk2gt_3,190,161,162).  visible(wk2gt_3,162). happensAt(active(wk2gt_3),162).
orientation(wk2gt_1,129,163). coords(wk2gt_1,115,53,163).  visible(wk2gt_1,163). happensAt(inactive(wk2gt_1),163). orientation(wk2gt_3,145,163). coords(wk2gt_3,191,161,163).  visible(wk2gt_3,163). happensAt(active(wk2gt_3),163).
orientation(wk2gt_1,129,164). coords(wk2gt_1,115,53,164).  visible(wk2gt_1,164). happensAt(inactive(wk2gt_1),164). orientation(wk2gt_3,145,164). coords(wk2gt_3,193,163,164).  visible(wk2gt_3,164). happensAt(active(wk2gt_3),164).
orientation(wk2gt_1,129,165). coords(wk2gt_1,115,53,165).  visible(wk2gt_1,165). happensAt(inactive(wk2gt_1),165). orientation(wk2gt_3,145,165). coords(wk2gt_3,195,165,165).  visible(wk2gt_3,165). happensAt(active(wk2gt_3),165).
orientation(wk2gt_1,129,166). coords(wk2gt_1,115,53,166).  visible(wk2gt_1,166). happensAt(inactive(wk2gt_1),166). orientation(wk2gt_3,145,166). coords(wk2gt_3,196,166,166).  visible(wk2gt_3,166). happensAt(active(wk2gt_3),166).
#show holdsAt/2.