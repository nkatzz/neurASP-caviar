
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

	orientation(mws1gt_4,0,130). coords(mws1gt_4,27,141,130).  visible(mws1gt_4,130). happensAt(walking(mws1gt_4),130). orientation(mws1gt_5,142,130). coords(mws1gt_5,77,64,130).  happensAt(appear(mws1gt_5),130). happensAt(walking(mws1gt_5),130).
orientation(mws1gt_4,0,131). coords(mws1gt_4,27,141,131).  visible(mws1gt_4,131). happensAt(walking(mws1gt_4),131). orientation(mws1gt_5,142,131). coords(mws1gt_5,77,64,131).  visible(mws1gt_5,131). happensAt(walking(mws1gt_5),131).
orientation(mws1gt_4,0,132). coords(mws1gt_4,27,141,132).  visible(mws1gt_4,132). happensAt(walking(mws1gt_4),132). orientation(mws1gt_5,142,132). coords(mws1gt_5,76,66,132).  visible(mws1gt_5,132). happensAt(walking(mws1gt_5),132).
orientation(mws1gt_4,0,133). coords(mws1gt_4,27,141,133).  visible(mws1gt_4,133). happensAt(walking(mws1gt_4),133). orientation(mws1gt_5,142,133). coords(mws1gt_5,76,66,133).  visible(mws1gt_5,133). happensAt(walking(mws1gt_5),133).
orientation(mws1gt_4,0,134). coords(mws1gt_4,27,140,134).  visible(mws1gt_4,134). happensAt(walking(mws1gt_4),134). orientation(mws1gt_5,142,134). coords(mws1gt_5,76,67,134).  visible(mws1gt_5,134). happensAt(walking(mws1gt_5),134).
orientation(mws1gt_4,0,135). coords(mws1gt_4,27,140,135).  visible(mws1gt_4,135). happensAt(walking(mws1gt_4),135). orientation(mws1gt_5,145,135). coords(mws1gt_5,75,68,135).  visible(mws1gt_5,135). happensAt(walking(mws1gt_5),135).
orientation(mws1gt_4,0,136). coords(mws1gt_4,27,140,136).  visible(mws1gt_4,136). happensAt(walking(mws1gt_4),136). orientation(mws1gt_5,145,136). coords(mws1gt_5,75,68,136).  visible(mws1gt_5,136). happensAt(walking(mws1gt_5),136).
orientation(mws1gt_4,0,137). coords(mws1gt_4,27,138,137).  visible(mws1gt_4,137). happensAt(walking(mws1gt_4),137). orientation(mws1gt_5,145,137). coords(mws1gt_5,75,69,137).  visible(mws1gt_5,137). happensAt(walking(mws1gt_5),137).
orientation(mws1gt_4,0,138). coords(mws1gt_4,27,138,138).  visible(mws1gt_4,138). happensAt(walking(mws1gt_4),138). orientation(mws1gt_5,145,138). coords(mws1gt_5,75,69,138).  visible(mws1gt_5,138). happensAt(walking(mws1gt_5),138).
orientation(mws1gt_4,0,139). coords(mws1gt_4,27,137,139).  visible(mws1gt_4,139). happensAt(walking(mws1gt_4),139). orientation(mws1gt_5,145,139). coords(mws1gt_5,75,69,139).  visible(mws1gt_5,139). happensAt(walking(mws1gt_5),139).
orientation(mws1gt_4,0,140). coords(mws1gt_4,28,136,140).  visible(mws1gt_4,140). happensAt(walking(mws1gt_4),140). orientation(mws1gt_5,145,140). coords(mws1gt_5,75,70,140).  visible(mws1gt_5,140). happensAt(walking(mws1gt_5),140).
orientation(mws1gt_4,0,141). coords(mws1gt_4,28,136,141).  visible(mws1gt_4,141). happensAt(walking(mws1gt_4),141). orientation(mws1gt_5,145,141). coords(mws1gt_5,75,70,141).  visible(mws1gt_5,141). happensAt(walking(mws1gt_5),141).
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
#show holdsAt/2.