
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

	orientation(mws1gt_1,155,130). coords(mws1gt_1,115,157,130).  visible(mws1gt_1,130). happensAt(walking(mws1gt_1),130). orientation(mws1gt_5,142,130). coords(mws1gt_5,77,64,130).  happensAt(appear(mws1gt_5),130). happensAt(walking(mws1gt_5),130).
orientation(mws1gt_1,155,131). coords(mws1gt_1,118,156,131).  visible(mws1gt_1,131). happensAt(walking(mws1gt_1),131). orientation(mws1gt_5,142,131). coords(mws1gt_5,77,64,131).  visible(mws1gt_5,131). happensAt(walking(mws1gt_5),131).
orientation(mws1gt_1,155,132). coords(mws1gt_1,120,156,132).  visible(mws1gt_1,132). happensAt(walking(mws1gt_1),132). orientation(mws1gt_5,142,132). coords(mws1gt_5,76,66,132).  visible(mws1gt_5,132). happensAt(walking(mws1gt_5),132).
orientation(mws1gt_1,155,133). coords(mws1gt_1,122,155,133).  visible(mws1gt_1,133). happensAt(walking(mws1gt_1),133). orientation(mws1gt_5,142,133). coords(mws1gt_5,76,66,133).  visible(mws1gt_5,133). happensAt(walking(mws1gt_5),133).
orientation(mws1gt_1,155,134). coords(mws1gt_1,125,155,134).  visible(mws1gt_1,134). happensAt(walking(mws1gt_1),134). orientation(mws1gt_5,142,134). coords(mws1gt_5,76,67,134).  visible(mws1gt_5,134). happensAt(walking(mws1gt_5),134).
orientation(mws1gt_1,155,135). coords(mws1gt_1,125,154,135).  visible(mws1gt_1,135). happensAt(walking(mws1gt_1),135). orientation(mws1gt_5,145,135). coords(mws1gt_5,75,68,135).  visible(mws1gt_5,135). happensAt(walking(mws1gt_5),135).
orientation(mws1gt_1,155,136). coords(mws1gt_1,127,154,136).  visible(mws1gt_1,136). happensAt(walking(mws1gt_1),136). orientation(mws1gt_5,145,136). coords(mws1gt_5,75,68,136).  visible(mws1gt_5,136). happensAt(walking(mws1gt_5),136).
orientation(mws1gt_1,155,137). coords(mws1gt_1,128,154,137).  visible(mws1gt_1,137). happensAt(walking(mws1gt_1),137). orientation(mws1gt_5,145,137). coords(mws1gt_5,75,69,137).  visible(mws1gt_5,137). happensAt(walking(mws1gt_5),137).
orientation(mws1gt_1,155,138). coords(mws1gt_1,129,153,138).  visible(mws1gt_1,138). happensAt(walking(mws1gt_1),138). orientation(mws1gt_5,145,138). coords(mws1gt_5,75,69,138).  visible(mws1gt_5,138). happensAt(walking(mws1gt_5),138).
orientation(mws1gt_1,155,139). coords(mws1gt_1,129,153,139).  visible(mws1gt_1,139). happensAt(walking(mws1gt_1),139). orientation(mws1gt_5,145,139). coords(mws1gt_5,75,69,139).  visible(mws1gt_5,139). happensAt(walking(mws1gt_5),139).
orientation(mws1gt_1,155,140). coords(mws1gt_1,129,153,140).  visible(mws1gt_1,140). happensAt(walking(mws1gt_1),140). orientation(mws1gt_5,145,140). coords(mws1gt_5,75,70,140).  visible(mws1gt_5,140). happensAt(walking(mws1gt_5),140).
orientation(mws1gt_1,155,141). coords(mws1gt_1,130,152,141).  visible(mws1gt_1,141). happensAt(walking(mws1gt_1),141). orientation(mws1gt_5,145,141). coords(mws1gt_5,75,70,141).  visible(mws1gt_5,141). happensAt(walking(mws1gt_5),141).
orientation(mws1gt_1,155,142). coords(mws1gt_1,130,152,142).  visible(mws1gt_1,142). happensAt(walking(mws1gt_1),142). orientation(mws1gt_5,145,142). coords(mws1gt_5,75,71,142).  visible(mws1gt_5,142). happensAt(walking(mws1gt_5),142).
orientation(mws1gt_1,155,143). coords(mws1gt_1,130,151,143).  visible(mws1gt_1,143). happensAt(walking(mws1gt_1),143). orientation(mws1gt_5,145,143). coords(mws1gt_5,75,71,143).  visible(mws1gt_5,143). happensAt(walking(mws1gt_5),143).
orientation(mws1gt_1,155,144). coords(mws1gt_1,131,151,144).  visible(mws1gt_1,144). happensAt(walking(mws1gt_1),144). orientation(mws1gt_5,145,144). coords(mws1gt_5,74,71,144).  visible(mws1gt_5,144). happensAt(walking(mws1gt_5),144).
orientation(mws1gt_1,155,145). coords(mws1gt_1,132,151,145).  visible(mws1gt_1,145). happensAt(walking(mws1gt_1),145). orientation(mws1gt_5,145,145). coords(mws1gt_5,74,72,145).  visible(mws1gt_5,145). happensAt(walking(mws1gt_5),145).
orientation(mws1gt_1,155,146). coords(mws1gt_1,134,151,146).  visible(mws1gt_1,146). happensAt(walking(mws1gt_1),146). orientation(mws1gt_5,145,146). coords(mws1gt_5,73,72,146).  visible(mws1gt_5,146). happensAt(walking(mws1gt_5),146).
orientation(mws1gt_1,155,147). coords(mws1gt_1,135,151,147).  visible(mws1gt_1,147). happensAt(walking(mws1gt_1),147). orientation(mws1gt_5,145,147). coords(mws1gt_5,72,73,147).  visible(mws1gt_5,147). happensAt(walking(mws1gt_5),147).
orientation(mws1gt_1,155,148). coords(mws1gt_1,136,151,148).  visible(mws1gt_1,148). happensAt(walking(mws1gt_1),148). orientation(mws1gt_5,145,148). coords(mws1gt_5,71,74,148).  visible(mws1gt_5,148). happensAt(walking(mws1gt_5),148).
orientation(mws1gt_1,155,149). coords(mws1gt_1,137,150,149).  visible(mws1gt_1,149). happensAt(walking(mws1gt_1),149). orientation(mws1gt_5,145,149). coords(mws1gt_5,71,75,149).  visible(mws1gt_5,149). happensAt(walking(mws1gt_5),149).
orientation(mws1gt_1,155,150). coords(mws1gt_1,138,150,150).  visible(mws1gt_1,150). happensAt(walking(mws1gt_1),150). orientation(mws1gt_5,145,150). coords(mws1gt_5,71,77,150).  visible(mws1gt_5,150). happensAt(walking(mws1gt_5),150).
orientation(mws1gt_1,155,151). coords(mws1gt_1,139,149,151).  visible(mws1gt_1,151). happensAt(walking(mws1gt_1),151). orientation(mws1gt_5,145,151). coords(mws1gt_5,71,78,151).  visible(mws1gt_5,151). happensAt(walking(mws1gt_5),151).
orientation(mws1gt_1,155,152). coords(mws1gt_1,139,149,152).  visible(mws1gt_1,152). happensAt(walking(mws1gt_1),152). orientation(mws1gt_5,145,152). coords(mws1gt_5,71,79,152).  visible(mws1gt_5,152). happensAt(walking(mws1gt_5),152).
orientation(mws1gt_1,155,153). coords(mws1gt_1,140,148,153).  visible(mws1gt_1,153). happensAt(walking(mws1gt_1),153). orientation(mws1gt_5,145,153). coords(mws1gt_5,71,80,153).  visible(mws1gt_5,153). happensAt(walking(mws1gt_5),153).
orientation(mws1gt_1,155,154). coords(mws1gt_1,140,148,154).  visible(mws1gt_1,154). happensAt(walking(mws1gt_1),154). orientation(mws1gt_5,145,154). coords(mws1gt_5,71,81,154).  visible(mws1gt_5,154). happensAt(walking(mws1gt_5),154).
#show holdsAt/2.