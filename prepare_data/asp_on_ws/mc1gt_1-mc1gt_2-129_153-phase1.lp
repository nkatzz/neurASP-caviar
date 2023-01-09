
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

	orientation(mc1gt_1,145,129). coords(mc1gt_1,110,82,129).  visible(mc1gt_1,129). happensAt(walking(mc1gt_1),129). orientation(mc1gt_2,144,129). coords(mc1gt_2,76,87,129).  visible(mc1gt_2,129). happensAt(walking(mc1gt_2),129).
orientation(mc1gt_1,145,130). coords(mc1gt_1,111,82,130).  visible(mc1gt_1,130). happensAt(walking(mc1gt_1),130). orientation(mc1gt_2,144,130). coords(mc1gt_2,76,88,130).  visible(mc1gt_2,130). happensAt(walking(mc1gt_2),130).
orientation(mc1gt_1,145,131). coords(mc1gt_1,111,81,131).  visible(mc1gt_1,131). happensAt(walking(mc1gt_1),131). orientation(mc1gt_2,144,131). coords(mc1gt_2,76,88,131).  visible(mc1gt_2,131). happensAt(walking(mc1gt_2),131).
orientation(mc1gt_1,145,132). coords(mc1gt_1,112,81,132).  visible(mc1gt_1,132). happensAt(walking(mc1gt_1),132). orientation(mc1gt_2,144,132). coords(mc1gt_2,76,88,132).  visible(mc1gt_2,132). happensAt(walking(mc1gt_2),132).
orientation(mc1gt_1,145,133). coords(mc1gt_1,112,81,133).  visible(mc1gt_1,133). happensAt(walking(mc1gt_1),133). orientation(mc1gt_2,144,133). coords(mc1gt_2,76,89,133).  visible(mc1gt_2,133). happensAt(walking(mc1gt_2),133).
orientation(mc1gt_1,145,134). coords(mc1gt_1,112,81,134).  visible(mc1gt_1,134). happensAt(walking(mc1gt_1),134). orientation(mc1gt_2,144,134). coords(mc1gt_2,76,89,134).  visible(mc1gt_2,134). happensAt(walking(mc1gt_2),134).
orientation(mc1gt_1,145,135). coords(mc1gt_1,113,81,135).  visible(mc1gt_1,135). happensAt(walking(mc1gt_1),135). orientation(mc1gt_2,144,135). coords(mc1gt_2,76,90,135).  visible(mc1gt_2,135). happensAt(walking(mc1gt_2),135).
orientation(mc1gt_1,145,136). coords(mc1gt_1,113,81,136).  visible(mc1gt_1,136). happensAt(walking(mc1gt_1),136). orientation(mc1gt_2,144,136). coords(mc1gt_2,76,90,136).  visible(mc1gt_2,136). happensAt(walking(mc1gt_2),136).
orientation(mc1gt_1,145,137). coords(mc1gt_1,113,81,137).  visible(mc1gt_1,137). happensAt(walking(mc1gt_1),137). orientation(mc1gt_2,144,137). coords(mc1gt_2,77,90,137).  visible(mc1gt_2,137). happensAt(walking(mc1gt_2),137).
orientation(mc1gt_1,145,138). coords(mc1gt_1,114,82,138).  visible(mc1gt_1,138). happensAt(walking(mc1gt_1),138). orientation(mc1gt_2,144,138). coords(mc1gt_2,77,91,138).  visible(mc1gt_2,138). happensAt(walking(mc1gt_2),138).
orientation(mc1gt_1,145,139). coords(mc1gt_1,116,82,139).  visible(mc1gt_1,139). happensAt(walking(mc1gt_1),139). orientation(mc1gt_2,144,139). coords(mc1gt_2,77,92,139).  visible(mc1gt_2,139). happensAt(walking(mc1gt_2),139).
orientation(mc1gt_1,145,140). coords(mc1gt_1,116,82,140).  visible(mc1gt_1,140). happensAt(walking(mc1gt_1),140). orientation(mc1gt_2,144,140). coords(mc1gt_2,76,92,140).  visible(mc1gt_2,140). happensAt(walking(mc1gt_2),140).
orientation(mc1gt_1,145,141). coords(mc1gt_1,117,82,141).  visible(mc1gt_1,141). happensAt(walking(mc1gt_1),141). orientation(mc1gt_2,144,141). coords(mc1gt_2,75,93,141).  visible(mc1gt_2,141). happensAt(walking(mc1gt_2),141).
orientation(mc1gt_1,145,142). coords(mc1gt_1,118,83,142).  visible(mc1gt_1,142). happensAt(walking(mc1gt_1),142). orientation(mc1gt_2,144,142). coords(mc1gt_2,75,94,142).  visible(mc1gt_2,142). happensAt(walking(mc1gt_2),142).
orientation(mc1gt_1,145,143). coords(mc1gt_1,118,83,143).  visible(mc1gt_1,143). happensAt(walking(mc1gt_1),143). orientation(mc1gt_2,144,143). coords(mc1gt_2,73,95,143).  visible(mc1gt_2,143). happensAt(walking(mc1gt_2),143).
orientation(mc1gt_1,145,144). coords(mc1gt_1,119,83,144).  visible(mc1gt_1,144). happensAt(walking(mc1gt_1),144). orientation(mc1gt_2,144,144). coords(mc1gt_2,73,95,144).  visible(mc1gt_2,144). happensAt(walking(mc1gt_2),144).
orientation(mc1gt_1,145,145). coords(mc1gt_1,119,83,145).  visible(mc1gt_1,145). happensAt(walking(mc1gt_1),145). orientation(mc1gt_2,144,145). coords(mc1gt_2,74,95,145).  visible(mc1gt_2,145). happensAt(walking(mc1gt_2),145).
orientation(mc1gt_1,145,146). coords(mc1gt_1,119,83,146).  visible(mc1gt_1,146). happensAt(walking(mc1gt_1),146). orientation(mc1gt_2,144,146). coords(mc1gt_2,74,95,146).  visible(mc1gt_2,146). happensAt(walking(mc1gt_2),146).
orientation(mc1gt_1,145,147). coords(mc1gt_1,120,83,147).  visible(mc1gt_1,147). happensAt(walking(mc1gt_1),147). orientation(mc1gt_2,144,147). coords(mc1gt_2,74,96,147).  visible(mc1gt_2,147). happensAt(walking(mc1gt_2),147).
orientation(mc1gt_1,145,148). coords(mc1gt_1,120,83,148).  visible(mc1gt_1,148). happensAt(walking(mc1gt_1),148). orientation(mc1gt_2,144,148). coords(mc1gt_2,75,96,148).  visible(mc1gt_2,148). happensAt(walking(mc1gt_2),148).
orientation(mc1gt_1,145,149). coords(mc1gt_1,120,83,149).  visible(mc1gt_1,149). happensAt(walking(mc1gt_1),149). orientation(mc1gt_2,144,149). coords(mc1gt_2,75,98,149).  visible(mc1gt_2,149). happensAt(walking(mc1gt_2),149).
orientation(mc1gt_1,145,150). coords(mc1gt_1,120,83,150).  visible(mc1gt_1,150). happensAt(walking(mc1gt_1),150). orientation(mc1gt_2,144,150). coords(mc1gt_2,78,101,150).  visible(mc1gt_2,150). happensAt(walking(mc1gt_2),150).
orientation(mc1gt_1,145,151). coords(mc1gt_1,121,83,151).  visible(mc1gt_1,151). happensAt(walking(mc1gt_1),151). orientation(mc1gt_2,144,151). coords(mc1gt_2,78,102,151).  visible(mc1gt_2,151). happensAt(walking(mc1gt_2),151).
orientation(mc1gt_1,145,152). coords(mc1gt_1,123,83,152).  visible(mc1gt_1,152). happensAt(walking(mc1gt_1),152). orientation(mc1gt_2,144,152). coords(mc1gt_2,79,103,152).  visible(mc1gt_2,152). happensAt(walking(mc1gt_2),152).
orientation(mc1gt_1,145,153). coords(mc1gt_1,124,83,153).  visible(mc1gt_1,153). happensAt(walking(mc1gt_1),153). orientation(mc1gt_2,144,153). coords(mc1gt_2,79,104,153).  visible(mc1gt_2,153). happensAt(walking(mc1gt_2),153).
#show holdsAt/2.