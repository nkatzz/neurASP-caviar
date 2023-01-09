
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

	orientation(mws1gt_3,154,130). coords(mws1gt_3,205,179,130).  visible(mws1gt_3,130). happensAt(walking(mws1gt_3),130). orientation(mws1gt_5,142,130). coords(mws1gt_5,77,64,130).  happensAt(appear(mws1gt_5),130). happensAt(walking(mws1gt_5),130).
orientation(mws1gt_3,154,131). coords(mws1gt_3,205,179,131).  visible(mws1gt_3,131). happensAt(walking(mws1gt_3),131). orientation(mws1gt_5,142,131). coords(mws1gt_5,77,64,131).  visible(mws1gt_5,131). happensAt(walking(mws1gt_5),131).
orientation(mws1gt_3,154,132). coords(mws1gt_3,204,178,132).  visible(mws1gt_3,132). happensAt(walking(mws1gt_3),132). orientation(mws1gt_5,142,132). coords(mws1gt_5,76,66,132).  visible(mws1gt_5,132). happensAt(walking(mws1gt_5),132).
orientation(mws1gt_3,154,133). coords(mws1gt_3,203,177,133).  visible(mws1gt_3,133). happensAt(walking(mws1gt_3),133). orientation(mws1gt_5,142,133). coords(mws1gt_5,76,66,133).  visible(mws1gt_5,133). happensAt(walking(mws1gt_5),133).
orientation(mws1gt_3,154,134). coords(mws1gt_3,203,177,134).  visible(mws1gt_3,134). happensAt(walking(mws1gt_3),134). orientation(mws1gt_5,142,134). coords(mws1gt_5,76,67,134).  visible(mws1gt_5,134). happensAt(walking(mws1gt_5),134).
orientation(mws1gt_3,154,135). coords(mws1gt_3,202,176,135).  visible(mws1gt_3,135). happensAt(walking(mws1gt_3),135). orientation(mws1gt_5,145,135). coords(mws1gt_5,75,68,135).  visible(mws1gt_5,135). happensAt(walking(mws1gt_5),135).
orientation(mws1gt_3,154,136). coords(mws1gt_3,201,175,136).  visible(mws1gt_3,136). happensAt(walking(mws1gt_3),136). orientation(mws1gt_5,145,136). coords(mws1gt_5,75,68,136).  visible(mws1gt_5,136). happensAt(walking(mws1gt_5),136).
orientation(mws1gt_3,154,137). coords(mws1gt_3,199,173,137).  visible(mws1gt_3,137). happensAt(walking(mws1gt_3),137). orientation(mws1gt_5,145,137). coords(mws1gt_5,75,69,137).  visible(mws1gt_5,137). happensAt(walking(mws1gt_5),137).
orientation(mws1gt_3,154,138). coords(mws1gt_3,197,171,138).  visible(mws1gt_3,138). happensAt(walking(mws1gt_3),138). orientation(mws1gt_5,145,138). coords(mws1gt_5,75,69,138).  visible(mws1gt_5,138). happensAt(walking(mws1gt_5),138).
orientation(mws1gt_3,154,139). coords(mws1gt_3,195,170,139).  visible(mws1gt_3,139). happensAt(walking(mws1gt_3),139). orientation(mws1gt_5,145,139). coords(mws1gt_5,75,69,139).  visible(mws1gt_5,139). happensAt(walking(mws1gt_5),139).
orientation(mws1gt_3,154,140). coords(mws1gt_3,193,168,140).  visible(mws1gt_3,140). happensAt(walking(mws1gt_3),140). orientation(mws1gt_5,145,140). coords(mws1gt_5,75,70,140).  visible(mws1gt_5,140). happensAt(walking(mws1gt_5),140).
orientation(mws1gt_3,154,141). coords(mws1gt_3,190,166,141).  visible(mws1gt_3,141). happensAt(walking(mws1gt_3),141). orientation(mws1gt_5,145,141). coords(mws1gt_5,75,70,141).  visible(mws1gt_5,141). happensAt(walking(mws1gt_5),141).
orientation(mws1gt_3,154,142). coords(mws1gt_3,188,164,142).  visible(mws1gt_3,142). happensAt(walking(mws1gt_3),142). orientation(mws1gt_5,145,142). coords(mws1gt_5,75,71,142).  visible(mws1gt_5,142). happensAt(walking(mws1gt_5),142).
orientation(mws1gt_3,154,143). coords(mws1gt_3,187,164,143).  visible(mws1gt_3,143). happensAt(walking(mws1gt_3),143). orientation(mws1gt_5,145,143). coords(mws1gt_5,75,71,143).  visible(mws1gt_5,143). happensAt(walking(mws1gt_5),143).
orientation(mws1gt_3,154,144). coords(mws1gt_3,186,163,144).  visible(mws1gt_3,144). happensAt(walking(mws1gt_3),144). orientation(mws1gt_5,145,144). coords(mws1gt_5,74,71,144).  visible(mws1gt_5,144). happensAt(walking(mws1gt_5),144).
orientation(mws1gt_3,154,145). coords(mws1gt_3,185,163,145).  visible(mws1gt_3,145). happensAt(walking(mws1gt_3),145). orientation(mws1gt_5,145,145). coords(mws1gt_5,74,72,145).  visible(mws1gt_5,145). happensAt(walking(mws1gt_5),145).
orientation(mws1gt_3,154,146). coords(mws1gt_3,185,162,146).  visible(mws1gt_3,146). happensAt(walking(mws1gt_3),146). orientation(mws1gt_5,145,146). coords(mws1gt_5,73,72,146).  visible(mws1gt_5,146). happensAt(walking(mws1gt_5),146).
orientation(mws1gt_3,154,147). coords(mws1gt_3,185,162,147).  visible(mws1gt_3,147). happensAt(walking(mws1gt_3),147). orientation(mws1gt_5,145,147). coords(mws1gt_5,72,73,147).  visible(mws1gt_5,147). happensAt(walking(mws1gt_5),147).
orientation(mws1gt_3,154,148). coords(mws1gt_3,184,161,148).  visible(mws1gt_3,148). happensAt(walking(mws1gt_3),148). orientation(mws1gt_5,145,148). coords(mws1gt_5,71,74,148).  visible(mws1gt_5,148). happensAt(walking(mws1gt_5),148).
orientation(mws1gt_3,154,149). coords(mws1gt_3,184,161,149).  visible(mws1gt_3,149). happensAt(walking(mws1gt_3),149). orientation(mws1gt_5,145,149). coords(mws1gt_5,71,75,149).  visible(mws1gt_5,149). happensAt(walking(mws1gt_5),149).
orientation(mws1gt_3,154,150). coords(mws1gt_3,183,160,150).  visible(mws1gt_3,150). happensAt(walking(mws1gt_3),150). orientation(mws1gt_5,145,150). coords(mws1gt_5,71,77,150).  visible(mws1gt_5,150). happensAt(walking(mws1gt_5),150).
orientation(mws1gt_3,154,151). coords(mws1gt_3,181,160,151).  visible(mws1gt_3,151). happensAt(walking(mws1gt_3),151). orientation(mws1gt_5,145,151). coords(mws1gt_5,71,78,151).  visible(mws1gt_5,151). happensAt(walking(mws1gt_5),151).
orientation(mws1gt_3,154,152). coords(mws1gt_3,179,158,152).  visible(mws1gt_3,152). happensAt(walking(mws1gt_3),152). orientation(mws1gt_5,145,152). coords(mws1gt_5,71,79,152).  visible(mws1gt_5,152). happensAt(walking(mws1gt_5),152).
orientation(mws1gt_3,154,153). coords(mws1gt_3,178,157,153).  visible(mws1gt_3,153). happensAt(walking(mws1gt_3),153). orientation(mws1gt_5,145,153). coords(mws1gt_5,71,80,153).  visible(mws1gt_5,153). happensAt(walking(mws1gt_5),153).
orientation(mws1gt_3,154,154). coords(mws1gt_3,176,155,154).  visible(mws1gt_3,154). happensAt(walking(mws1gt_3),154). orientation(mws1gt_5,145,154). coords(mws1gt_5,71,81,154).  visible(mws1gt_5,154). happensAt(walking(mws1gt_5),154).
#show holdsAt/2.