
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

	orientation(mws1gt_3,150,126). coords(mws1gt_3,208,187,126).  visible(mws1gt_3,126). happensAt(walking(mws1gt_3),126). orientation(mws1gt_4,0,126). coords(mws1gt_4,27,144,126).  visible(mws1gt_4,126). happensAt(walking(mws1gt_4),126).
orientation(mws1gt_3,150,127). coords(mws1gt_3,207,184,127).  visible(mws1gt_3,127). happensAt(walking(mws1gt_3),127). orientation(mws1gt_4,0,127). coords(mws1gt_4,27,142,127).  visible(mws1gt_4,127). happensAt(walking(mws1gt_4),127).
orientation(mws1gt_3,150,128). coords(mws1gt_3,206,182,128).  visible(mws1gt_3,128). happensAt(walking(mws1gt_3),128). orientation(mws1gt_4,0,128). coords(mws1gt_4,27,142,128).  visible(mws1gt_4,128). happensAt(walking(mws1gt_4),128).
orientation(mws1gt_3,150,129). coords(mws1gt_3,206,180,129).  visible(mws1gt_3,129). happensAt(walking(mws1gt_3),129). orientation(mws1gt_4,0,129). coords(mws1gt_4,27,142,129).  visible(mws1gt_4,129). happensAt(walking(mws1gt_4),129).
orientation(mws1gt_3,154,130). coords(mws1gt_3,205,179,130).  visible(mws1gt_3,130). happensAt(walking(mws1gt_3),130). orientation(mws1gt_4,0,130). coords(mws1gt_4,27,141,130).  visible(mws1gt_4,130). happensAt(walking(mws1gt_4),130).
orientation(mws1gt_3,154,131). coords(mws1gt_3,205,179,131).  visible(mws1gt_3,131). happensAt(walking(mws1gt_3),131). orientation(mws1gt_4,0,131). coords(mws1gt_4,27,141,131).  visible(mws1gt_4,131). happensAt(walking(mws1gt_4),131).
orientation(mws1gt_3,154,132). coords(mws1gt_3,204,178,132).  visible(mws1gt_3,132). happensAt(walking(mws1gt_3),132). orientation(mws1gt_4,0,132). coords(mws1gt_4,27,141,132).  visible(mws1gt_4,132). happensAt(walking(mws1gt_4),132).
orientation(mws1gt_3,154,133). coords(mws1gt_3,203,177,133).  visible(mws1gt_3,133). happensAt(walking(mws1gt_3),133). orientation(mws1gt_4,0,133). coords(mws1gt_4,27,141,133).  visible(mws1gt_4,133). happensAt(walking(mws1gt_4),133).
orientation(mws1gt_3,154,134). coords(mws1gt_3,203,177,134).  visible(mws1gt_3,134). happensAt(walking(mws1gt_3),134). orientation(mws1gt_4,0,134). coords(mws1gt_4,27,140,134).  visible(mws1gt_4,134). happensAt(walking(mws1gt_4),134).
orientation(mws1gt_3,154,135). coords(mws1gt_3,202,176,135).  visible(mws1gt_3,135). happensAt(walking(mws1gt_3),135). orientation(mws1gt_4,0,135). coords(mws1gt_4,27,140,135).  visible(mws1gt_4,135). happensAt(walking(mws1gt_4),135).
orientation(mws1gt_3,154,136). coords(mws1gt_3,201,175,136).  visible(mws1gt_3,136). happensAt(walking(mws1gt_3),136). orientation(mws1gt_4,0,136). coords(mws1gt_4,27,140,136).  visible(mws1gt_4,136). happensAt(walking(mws1gt_4),136).
orientation(mws1gt_3,154,137). coords(mws1gt_3,199,173,137).  visible(mws1gt_3,137). happensAt(walking(mws1gt_3),137). orientation(mws1gt_4,0,137). coords(mws1gt_4,27,138,137).  visible(mws1gt_4,137). happensAt(walking(mws1gt_4),137).
orientation(mws1gt_3,154,138). coords(mws1gt_3,197,171,138).  visible(mws1gt_3,138). happensAt(walking(mws1gt_3),138). orientation(mws1gt_4,0,138). coords(mws1gt_4,27,138,138).  visible(mws1gt_4,138). happensAt(walking(mws1gt_4),138).
orientation(mws1gt_3,154,139). coords(mws1gt_3,195,170,139).  visible(mws1gt_3,139). happensAt(walking(mws1gt_3),139). orientation(mws1gt_4,0,139). coords(mws1gt_4,27,137,139).  visible(mws1gt_4,139). happensAt(walking(mws1gt_4),139).
orientation(mws1gt_3,154,140). coords(mws1gt_3,193,168,140).  visible(mws1gt_3,140). happensAt(walking(mws1gt_3),140). orientation(mws1gt_4,0,140). coords(mws1gt_4,28,136,140).  visible(mws1gt_4,140). happensAt(walking(mws1gt_4),140).
orientation(mws1gt_3,154,141). coords(mws1gt_3,190,166,141).  visible(mws1gt_3,141). happensAt(walking(mws1gt_3),141). orientation(mws1gt_4,0,141). coords(mws1gt_4,28,136,141).  visible(mws1gt_4,141). happensAt(walking(mws1gt_4),141).
orientation(mws1gt_3,154,142). coords(mws1gt_3,188,164,142).  visible(mws1gt_3,142). happensAt(walking(mws1gt_3),142). orientation(mws1gt_4,0,142). coords(mws1gt_4,28,134,142).  visible(mws1gt_4,142). happensAt(walking(mws1gt_4),142).
orientation(mws1gt_3,154,143). coords(mws1gt_3,187,164,143).  visible(mws1gt_3,143). happensAt(walking(mws1gt_3),143). orientation(mws1gt_4,0,143). coords(mws1gt_4,28,134,143).  visible(mws1gt_4,143). happensAt(walking(mws1gt_4),143).
orientation(mws1gt_3,154,144). coords(mws1gt_3,186,163,144).  visible(mws1gt_3,144). happensAt(walking(mws1gt_3),144). orientation(mws1gt_4,0,144). coords(mws1gt_4,28,134,144).  visible(mws1gt_4,144). happensAt(walking(mws1gt_4),144).
orientation(mws1gt_3,154,145). coords(mws1gt_3,185,163,145).  visible(mws1gt_3,145). happensAt(walking(mws1gt_3),145). orientation(mws1gt_4,0,145). coords(mws1gt_4,28,134,145).  visible(mws1gt_4,145). happensAt(walking(mws1gt_4),145).
orientation(mws1gt_3,154,146). coords(mws1gt_3,185,162,146).  visible(mws1gt_3,146). happensAt(walking(mws1gt_3),146). orientation(mws1gt_4,0,146). coords(mws1gt_4,28,134,146).  visible(mws1gt_4,146). happensAt(walking(mws1gt_4),146).
orientation(mws1gt_3,154,147). coords(mws1gt_3,185,162,147).  visible(mws1gt_3,147). happensAt(walking(mws1gt_3),147). orientation(mws1gt_4,0,147). coords(mws1gt_4,28,134,147).  visible(mws1gt_4,147). happensAt(walking(mws1gt_4),147).
orientation(mws1gt_3,154,148). coords(mws1gt_3,184,161,148).  visible(mws1gt_3,148). happensAt(walking(mws1gt_3),148). orientation(mws1gt_4,0,148). coords(mws1gt_4,28,133,148).  visible(mws1gt_4,148). happensAt(walking(mws1gt_4),148).
orientation(mws1gt_3,154,149). coords(mws1gt_3,184,161,149).  visible(mws1gt_3,149). happensAt(walking(mws1gt_3),149). orientation(mws1gt_4,0,149). coords(mws1gt_4,28,133,149).  visible(mws1gt_4,149). happensAt(walking(mws1gt_4),149).
orientation(mws1gt_3,154,150). coords(mws1gt_3,183,160,150).  visible(mws1gt_3,150). happensAt(walking(mws1gt_3),150). orientation(mws1gt_4,0,150). coords(mws1gt_4,28,133,150).  visible(mws1gt_4,150). happensAt(walking(mws1gt_4),150).
#show holdsAt/2.