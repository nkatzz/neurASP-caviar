
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

	orientation(mws1gt_1,155,126). coords(mws1gt_1,115,162,126).  visible(mws1gt_1,126). happensAt(walking(mws1gt_1),126). orientation(mws1gt_4,0,126). coords(mws1gt_4,27,144,126).  visible(mws1gt_4,126). happensAt(walking(mws1gt_4),126).
orientation(mws1gt_1,155,127). coords(mws1gt_1,115,162,127).  visible(mws1gt_1,127). happensAt(walking(mws1gt_1),127). orientation(mws1gt_4,0,127). coords(mws1gt_4,27,142,127).  visible(mws1gt_4,127). happensAt(walking(mws1gt_4),127).
orientation(mws1gt_1,155,128). coords(mws1gt_1,115,161,128).  visible(mws1gt_1,128). happensAt(walking(mws1gt_1),128). orientation(mws1gt_4,0,128). coords(mws1gt_4,27,142,128).  visible(mws1gt_4,128). happensAt(walking(mws1gt_4),128).
orientation(mws1gt_1,155,129). coords(mws1gt_1,115,159,129).  visible(mws1gt_1,129). happensAt(walking(mws1gt_1),129). orientation(mws1gt_4,0,129). coords(mws1gt_4,27,142,129).  visible(mws1gt_4,129). happensAt(walking(mws1gt_4),129).
orientation(mws1gt_1,155,130). coords(mws1gt_1,115,157,130).  visible(mws1gt_1,130). happensAt(walking(mws1gt_1),130). orientation(mws1gt_4,0,130). coords(mws1gt_4,27,141,130).  visible(mws1gt_4,130). happensAt(walking(mws1gt_4),130).
orientation(mws1gt_1,155,131). coords(mws1gt_1,118,156,131).  visible(mws1gt_1,131). happensAt(walking(mws1gt_1),131). orientation(mws1gt_4,0,131). coords(mws1gt_4,27,141,131).  visible(mws1gt_4,131). happensAt(walking(mws1gt_4),131).
orientation(mws1gt_1,155,132). coords(mws1gt_1,120,156,132).  visible(mws1gt_1,132). happensAt(walking(mws1gt_1),132). orientation(mws1gt_4,0,132). coords(mws1gt_4,27,141,132).  visible(mws1gt_4,132). happensAt(walking(mws1gt_4),132).
orientation(mws1gt_1,155,133). coords(mws1gt_1,122,155,133).  visible(mws1gt_1,133). happensAt(walking(mws1gt_1),133). orientation(mws1gt_4,0,133). coords(mws1gt_4,27,141,133).  visible(mws1gt_4,133). happensAt(walking(mws1gt_4),133).
orientation(mws1gt_1,155,134). coords(mws1gt_1,125,155,134).  visible(mws1gt_1,134). happensAt(walking(mws1gt_1),134). orientation(mws1gt_4,0,134). coords(mws1gt_4,27,140,134).  visible(mws1gt_4,134). happensAt(walking(mws1gt_4),134).
orientation(mws1gt_1,155,135). coords(mws1gt_1,125,154,135).  visible(mws1gt_1,135). happensAt(walking(mws1gt_1),135). orientation(mws1gt_4,0,135). coords(mws1gt_4,27,140,135).  visible(mws1gt_4,135). happensAt(walking(mws1gt_4),135).
orientation(mws1gt_1,155,136). coords(mws1gt_1,127,154,136).  visible(mws1gt_1,136). happensAt(walking(mws1gt_1),136). orientation(mws1gt_4,0,136). coords(mws1gt_4,27,140,136).  visible(mws1gt_4,136). happensAt(walking(mws1gt_4),136).
orientation(mws1gt_1,155,137). coords(mws1gt_1,128,154,137).  visible(mws1gt_1,137). happensAt(walking(mws1gt_1),137). orientation(mws1gt_4,0,137). coords(mws1gt_4,27,138,137).  visible(mws1gt_4,137). happensAt(walking(mws1gt_4),137).
orientation(mws1gt_1,155,138). coords(mws1gt_1,129,153,138).  visible(mws1gt_1,138). happensAt(walking(mws1gt_1),138). orientation(mws1gt_4,0,138). coords(mws1gt_4,27,138,138).  visible(mws1gt_4,138). happensAt(walking(mws1gt_4),138).
orientation(mws1gt_1,155,139). coords(mws1gt_1,129,153,139).  visible(mws1gt_1,139). happensAt(walking(mws1gt_1),139). orientation(mws1gt_4,0,139). coords(mws1gt_4,27,137,139).  visible(mws1gt_4,139). happensAt(walking(mws1gt_4),139).
orientation(mws1gt_1,155,140). coords(mws1gt_1,129,153,140).  visible(mws1gt_1,140). happensAt(walking(mws1gt_1),140). orientation(mws1gt_4,0,140). coords(mws1gt_4,28,136,140).  visible(mws1gt_4,140). happensAt(walking(mws1gt_4),140).
orientation(mws1gt_1,155,141). coords(mws1gt_1,130,152,141).  visible(mws1gt_1,141). happensAt(walking(mws1gt_1),141). orientation(mws1gt_4,0,141). coords(mws1gt_4,28,136,141).  visible(mws1gt_4,141). happensAt(walking(mws1gt_4),141).
orientation(mws1gt_1,155,142). coords(mws1gt_1,130,152,142).  visible(mws1gt_1,142). happensAt(walking(mws1gt_1),142). orientation(mws1gt_4,0,142). coords(mws1gt_4,28,134,142).  visible(mws1gt_4,142). happensAt(walking(mws1gt_4),142).
orientation(mws1gt_1,155,143). coords(mws1gt_1,130,151,143).  visible(mws1gt_1,143). happensAt(walking(mws1gt_1),143). orientation(mws1gt_4,0,143). coords(mws1gt_4,28,134,143).  visible(mws1gt_4,143). happensAt(walking(mws1gt_4),143).
orientation(mws1gt_1,155,144). coords(mws1gt_1,131,151,144).  visible(mws1gt_1,144). happensAt(walking(mws1gt_1),144). orientation(mws1gt_4,0,144). coords(mws1gt_4,28,134,144).  visible(mws1gt_4,144). happensAt(walking(mws1gt_4),144).
orientation(mws1gt_1,155,145). coords(mws1gt_1,132,151,145).  visible(mws1gt_1,145). happensAt(walking(mws1gt_1),145). orientation(mws1gt_4,0,145). coords(mws1gt_4,28,134,145).  visible(mws1gt_4,145). happensAt(walking(mws1gt_4),145).
orientation(mws1gt_1,155,146). coords(mws1gt_1,134,151,146).  visible(mws1gt_1,146). happensAt(walking(mws1gt_1),146). orientation(mws1gt_4,0,146). coords(mws1gt_4,28,134,146).  visible(mws1gt_4,146). happensAt(walking(mws1gt_4),146).
orientation(mws1gt_1,155,147). coords(mws1gt_1,135,151,147).  visible(mws1gt_1,147). happensAt(walking(mws1gt_1),147). orientation(mws1gt_4,0,147). coords(mws1gt_4,28,134,147).  visible(mws1gt_4,147). happensAt(walking(mws1gt_4),147).
orientation(mws1gt_1,155,148). coords(mws1gt_1,136,151,148).  visible(mws1gt_1,148). happensAt(walking(mws1gt_1),148). orientation(mws1gt_4,0,148). coords(mws1gt_4,28,133,148).  visible(mws1gt_4,148). happensAt(walking(mws1gt_4),148).
orientation(mws1gt_1,155,149). coords(mws1gt_1,137,150,149).  visible(mws1gt_1,149). happensAt(walking(mws1gt_1),149). orientation(mws1gt_4,0,149). coords(mws1gt_4,28,133,149).  visible(mws1gt_4,149). happensAt(walking(mws1gt_4),149).
orientation(mws1gt_1,155,150). coords(mws1gt_1,138,150,150).  visible(mws1gt_1,150). happensAt(walking(mws1gt_1),150). orientation(mws1gt_4,0,150). coords(mws1gt_4,28,133,150).  visible(mws1gt_4,150). happensAt(walking(mws1gt_4),150).
#show holdsAt/2.