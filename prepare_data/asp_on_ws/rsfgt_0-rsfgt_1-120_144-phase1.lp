
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

	orientation(rsfgt_0,160,120). coords(rsfgt_0,44,155,120).  visible(rsfgt_0,120). happensAt(walking(rsfgt_0),120). orientation(rsfgt_1,150,120). coords(rsfgt_1,166,106,120).  visible(rsfgt_1,120). happensAt(active(rsfgt_1),120).
orientation(rsfgt_0,160,121). coords(rsfgt_0,44,155,121).  visible(rsfgt_0,121). happensAt(walking(rsfgt_0),121). orientation(rsfgt_1,150,121). coords(rsfgt_1,166,106,121).  visible(rsfgt_1,121). happensAt(active(rsfgt_1),121).
orientation(rsfgt_0,160,122). coords(rsfgt_0,44,155,122).  visible(rsfgt_0,122). happensAt(walking(rsfgt_0),122). orientation(rsfgt_1,150,122). coords(rsfgt_1,166,106,122).  visible(rsfgt_1,122). happensAt(active(rsfgt_1),122).
orientation(rsfgt_0,160,123). coords(rsfgt_0,43,155,123).  visible(rsfgt_0,123). happensAt(walking(rsfgt_0),123). orientation(rsfgt_1,150,123). coords(rsfgt_1,166,106,123).  visible(rsfgt_1,123). happensAt(active(rsfgt_1),123).
orientation(rsfgt_0,160,124). coords(rsfgt_0,43,155,124).  visible(rsfgt_0,124). happensAt(walking(rsfgt_0),124). orientation(rsfgt_1,150,124). coords(rsfgt_1,166,107,124).  visible(rsfgt_1,124). happensAt(active(rsfgt_1),124).
orientation(rsfgt_0,160,125). coords(rsfgt_0,42,155,125).  visible(rsfgt_0,125). happensAt(walking(rsfgt_0),125). orientation(rsfgt_1,150,125). coords(rsfgt_1,166,107,125).  visible(rsfgt_1,125). happensAt(active(rsfgt_1),125).
orientation(rsfgt_0,160,126). coords(rsfgt_0,41,155,126).  visible(rsfgt_0,126). happensAt(walking(rsfgt_0),126). orientation(rsfgt_1,150,126). coords(rsfgt_1,166,108,126).  visible(rsfgt_1,126). happensAt(active(rsfgt_1),126).
orientation(rsfgt_0,160,127). coords(rsfgt_0,41,155,127).  visible(rsfgt_0,127). happensAt(walking(rsfgt_0),127). orientation(rsfgt_1,150,127). coords(rsfgt_1,166,108,127).  visible(rsfgt_1,127). happensAt(active(rsfgt_1),127).
orientation(rsfgt_0,160,128). coords(rsfgt_0,41,155,128).  visible(rsfgt_0,128). happensAt(walking(rsfgt_0),128). orientation(rsfgt_1,150,128). coords(rsfgt_1,166,108,128).  visible(rsfgt_1,128). happensAt(active(rsfgt_1),128).
orientation(rsfgt_0,160,129). coords(rsfgt_0,41,155,129).  visible(rsfgt_0,129). happensAt(walking(rsfgt_0),129). orientation(rsfgt_1,150,129). coords(rsfgt_1,166,108,129).  visible(rsfgt_1,129). happensAt(active(rsfgt_1),129).
orientation(rsfgt_0,160,130). coords(rsfgt_0,41,155,130).  visible(rsfgt_0,130). happensAt(walking(rsfgt_0),130). orientation(rsfgt_1,150,130). coords(rsfgt_1,166,108,130).  visible(rsfgt_1,130). happensAt(active(rsfgt_1),130).
orientation(rsfgt_0,160,131). coords(rsfgt_0,41,155,131).  visible(rsfgt_0,131). happensAt(walking(rsfgt_0),131). orientation(rsfgt_1,150,131). coords(rsfgt_1,166,108,131).  visible(rsfgt_1,131). happensAt(active(rsfgt_1),131).
orientation(rsfgt_0,160,132). coords(rsfgt_0,41,155,132).  visible(rsfgt_0,132). happensAt(walking(rsfgt_0),132). orientation(rsfgt_1,150,132). coords(rsfgt_1,166,108,132).  visible(rsfgt_1,132). happensAt(active(rsfgt_1),132).
orientation(rsfgt_0,160,133). coords(rsfgt_0,39,155,133).  visible(rsfgt_0,133). happensAt(walking(rsfgt_0),133). orientation(rsfgt_1,150,133). coords(rsfgt_1,166,108,133).  visible(rsfgt_1,133). happensAt(active(rsfgt_1),133).
orientation(rsfgt_0,160,134). coords(rsfgt_0,39,155,134).  visible(rsfgt_0,134). happensAt(walking(rsfgt_0),134). orientation(rsfgt_1,150,134). coords(rsfgt_1,166,108,134).  visible(rsfgt_1,134). happensAt(active(rsfgt_1),134).
orientation(rsfgt_0,160,135). coords(rsfgt_0,39,155,135).  visible(rsfgt_0,135). happensAt(walking(rsfgt_0),135). orientation(rsfgt_1,150,135). coords(rsfgt_1,166,108,135).  visible(rsfgt_1,135). happensAt(active(rsfgt_1),135).
orientation(rsfgt_0,160,136). coords(rsfgt_0,39,155,136).  visible(rsfgt_0,136). happensAt(walking(rsfgt_0),136). orientation(rsfgt_1,150,136). coords(rsfgt_1,166,108,136).  visible(rsfgt_1,136). happensAt(active(rsfgt_1),136).
orientation(rsfgt_0,160,137). coords(rsfgt_0,39,155,137).  visible(rsfgt_0,137). happensAt(walking(rsfgt_0),137). orientation(rsfgt_1,150,137). coords(rsfgt_1,166,108,137).  visible(rsfgt_1,137). happensAt(active(rsfgt_1),137).
orientation(rsfgt_0,160,138). coords(rsfgt_0,38,155,138).  visible(rsfgt_0,138). happensAt(walking(rsfgt_0),138). orientation(rsfgt_1,150,138). coords(rsfgt_1,166,108,138).  visible(rsfgt_1,138). happensAt(active(rsfgt_1),138).
orientation(rsfgt_0,160,139). coords(rsfgt_0,38,155,139).  visible(rsfgt_0,139). happensAt(walking(rsfgt_0),139). orientation(rsfgt_1,150,139). coords(rsfgt_1,166,108,139).  visible(rsfgt_1,139). happensAt(active(rsfgt_1),139).
orientation(rsfgt_0,160,140). coords(rsfgt_0,38,155,140).  visible(rsfgt_0,140). happensAt(walking(rsfgt_0),140). orientation(rsfgt_1,150,140). coords(rsfgt_1,166,108,140).  visible(rsfgt_1,140). happensAt(active(rsfgt_1),140).
orientation(rsfgt_0,160,141). coords(rsfgt_0,38,155,141).  visible(rsfgt_0,141). happensAt(walking(rsfgt_0),141). orientation(rsfgt_1,150,141). coords(rsfgt_1,166,108,141).  visible(rsfgt_1,141). happensAt(active(rsfgt_1),141).
orientation(rsfgt_0,160,142). coords(rsfgt_0,38,155,142).  visible(rsfgt_0,142). happensAt(walking(rsfgt_0),142). orientation(rsfgt_1,150,142). coords(rsfgt_1,168,111,142).  visible(rsfgt_1,142). happensAt(walking(rsfgt_1),142).
orientation(rsfgt_0,160,143). coords(rsfgt_0,38,155,143).  visible(rsfgt_0,143). happensAt(walking(rsfgt_0),143). orientation(rsfgt_1,150,143). coords(rsfgt_1,168,113,143).  visible(rsfgt_1,143). happensAt(walking(rsfgt_1),143).
orientation(rsfgt_0,160,144). coords(rsfgt_0,35,155,144).  visible(rsfgt_0,144). happensAt(walking(rsfgt_0),144). orientation(rsfgt_1,150,144). coords(rsfgt_1,171,113,144).  visible(rsfgt_1,144). happensAt(walking(rsfgt_1),144).
#show holdsAt/2.