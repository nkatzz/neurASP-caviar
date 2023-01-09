
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

	orientation(wk2gt_1,132,130). coords(wk2gt_1,114,52,130).  visible(wk2gt_1,130). happensAt(inactive(wk2gt_1),130). orientation(wk2gt_3,157,130). coords(wk2gt_3,189,172,130).  visible(wk2gt_3,130). happensAt(active(wk2gt_3),130).
orientation(wk2gt_1,132,131). coords(wk2gt_1,114,52,131).  visible(wk2gt_1,131). happensAt(inactive(wk2gt_1),131). orientation(wk2gt_3,154,131). coords(wk2gt_3,186,171,131).  visible(wk2gt_3,131). happensAt(active(wk2gt_3),131).
orientation(wk2gt_1,132,132). coords(wk2gt_1,114,52,132).  visible(wk2gt_1,132). happensAt(inactive(wk2gt_1),132). orientation(wk2gt_3,154,132). coords(wk2gt_3,184,169,132).  visible(wk2gt_3,132). happensAt(active(wk2gt_3),132).
orientation(wk2gt_1,132,133). coords(wk2gt_1,114,52,133).  visible(wk2gt_1,133). happensAt(inactive(wk2gt_1),133). orientation(wk2gt_3,154,133). coords(wk2gt_3,182,168,133).  visible(wk2gt_3,133). happensAt(active(wk2gt_3),133).
orientation(wk2gt_1,132,134). coords(wk2gt_1,114,52,134).  visible(wk2gt_1,134). happensAt(inactive(wk2gt_1),134). orientation(wk2gt_3,154,134). coords(wk2gt_3,182,166,134).  visible(wk2gt_3,134). happensAt(active(wk2gt_3),134).
orientation(wk2gt_1,132,135). coords(wk2gt_1,114,52,135).  visible(wk2gt_1,135). happensAt(inactive(wk2gt_1),135). orientation(wk2gt_3,154,135). coords(wk2gt_3,181,164,135).  visible(wk2gt_3,135). happensAt(active(wk2gt_3),135).
orientation(wk2gt_1,132,136). coords(wk2gt_1,114,52,136).  visible(wk2gt_1,136). happensAt(inactive(wk2gt_1),136). orientation(wk2gt_3,154,136). coords(wk2gt_3,181,162,136).  visible(wk2gt_3,136). happensAt(active(wk2gt_3),136).
orientation(wk2gt_1,132,137). coords(wk2gt_1,114,52,137).  visible(wk2gt_1,137). happensAt(inactive(wk2gt_1),137). orientation(wk2gt_3,154,137). coords(wk2gt_3,181,161,137).  visible(wk2gt_3,137). happensAt(active(wk2gt_3),137).
orientation(wk2gt_1,132,138). coords(wk2gt_1,114,52,138).  visible(wk2gt_1,138). happensAt(inactive(wk2gt_1),138). orientation(wk2gt_3,154,138). coords(wk2gt_3,180,159,138).  visible(wk2gt_3,138). happensAt(active(wk2gt_3),138).
orientation(wk2gt_1,132,139). coords(wk2gt_1,114,52,139).  visible(wk2gt_1,139). happensAt(inactive(wk2gt_1),139). orientation(wk2gt_3,154,139). coords(wk2gt_3,180,159,139).  visible(wk2gt_3,139). happensAt(active(wk2gt_3),139).
orientation(wk2gt_1,132,140). coords(wk2gt_1,114,52,140).  visible(wk2gt_1,140). happensAt(inactive(wk2gt_1),140). orientation(wk2gt_3,154,140). coords(wk2gt_3,179,159,140).  visible(wk2gt_3,140). happensAt(active(wk2gt_3),140).
orientation(wk2gt_1,132,141). coords(wk2gt_1,114,52,141).  visible(wk2gt_1,141). happensAt(inactive(wk2gt_1),141). orientation(wk2gt_3,154,141). coords(wk2gt_3,179,159,141).  visible(wk2gt_3,141). happensAt(active(wk2gt_3),141).
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
#show holdsAt/2.