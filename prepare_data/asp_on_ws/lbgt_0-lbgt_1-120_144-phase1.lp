
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

	orientation(lbgt_0,145,120). coords(lbgt_0,163,122,120).  visible(lbgt_0,120). happensAt(active(lbgt_0),120). orientation(lbgt_1,155,120). coords(lbgt_1,75,83,120).  visible(lbgt_1,120). happensAt(walking(lbgt_1),120).
orientation(lbgt_0,145,121). coords(lbgt_0,164,124,121).  visible(lbgt_0,121). happensAt(active(lbgt_0),121). orientation(lbgt_1,155,121). coords(lbgt_1,76,82,121).  visible(lbgt_1,121). happensAt(walking(lbgt_1),121).
orientation(lbgt_0,145,122). coords(lbgt_0,166,125,122).  visible(lbgt_0,122). happensAt(active(lbgt_0),122). orientation(lbgt_1,155,122). coords(lbgt_1,76,83,122).  visible(lbgt_1,122). happensAt(walking(lbgt_1),122).
orientation(lbgt_0,145,123). coords(lbgt_0,167,127,123).  visible(lbgt_0,123). happensAt(active(lbgt_0),123). orientation(lbgt_1,155,123). coords(lbgt_1,76,84,123).  visible(lbgt_1,123). happensAt(walking(lbgt_1),123).
orientation(lbgt_0,145,124). coords(lbgt_0,168,128,124).  visible(lbgt_0,124). happensAt(active(lbgt_0),124). orientation(lbgt_1,155,124). coords(lbgt_1,76,84,124).  visible(lbgt_1,124). happensAt(walking(lbgt_1),124).
orientation(lbgt_0,145,125). coords(lbgt_0,169,129,125).  visible(lbgt_0,125). happensAt(active(lbgt_0),125). orientation(lbgt_1,155,125). coords(lbgt_1,76,84,125).  visible(lbgt_1,125). happensAt(walking(lbgt_1),125).
orientation(lbgt_0,145,126). coords(lbgt_0,170,131,126).  visible(lbgt_0,126). happensAt(active(lbgt_0),126). orientation(lbgt_1,155,126). coords(lbgt_1,76,84,126).  visible(lbgt_1,126). happensAt(walking(lbgt_1),126).
orientation(lbgt_0,145,127). coords(lbgt_0,170,132,127).  visible(lbgt_0,127). happensAt(active(lbgt_0),127). orientation(lbgt_1,155,127). coords(lbgt_1,76,84,127).  visible(lbgt_1,127). happensAt(walking(lbgt_1),127).
orientation(lbgt_0,151,128). coords(lbgt_0,171,133,128).  visible(lbgt_0,128). happensAt(walking(lbgt_0),128). orientation(lbgt_1,155,128). coords(lbgt_1,76,84,128).  visible(lbgt_1,128). happensAt(walking(lbgt_1),128).
orientation(lbgt_0,155,129). coords(lbgt_0,171,134,129).  visible(lbgt_0,129). happensAt(walking(lbgt_0),129). orientation(lbgt_1,155,129). coords(lbgt_1,76,84,129).  visible(lbgt_1,129). happensAt(walking(lbgt_1),129).
orientation(lbgt_0,157,130). coords(lbgt_0,171,135,130).  visible(lbgt_0,130). happensAt(walking(lbgt_0),130). orientation(lbgt_1,155,130). coords(lbgt_1,76,85,130).  visible(lbgt_1,130). happensAt(walking(lbgt_1),130).
orientation(lbgt_0,159,131). coords(lbgt_0,172,136,131).  visible(lbgt_0,131). happensAt(walking(lbgt_0),131). orientation(lbgt_1,155,131). coords(lbgt_1,76,85,131).  visible(lbgt_1,131). happensAt(walking(lbgt_1),131).
orientation(lbgt_0,159,132). coords(lbgt_0,174,137,132).  visible(lbgt_0,132). happensAt(walking(lbgt_0),132). orientation(lbgt_1,155,132). coords(lbgt_1,76,86,132).  visible(lbgt_1,132). happensAt(walking(lbgt_1),132).
orientation(lbgt_0,170,133). coords(lbgt_0,177,138,133).  visible(lbgt_0,133). happensAt(walking(lbgt_0),133). orientation(lbgt_1,155,133). coords(lbgt_1,76,86,133).  visible(lbgt_1,133). happensAt(walking(lbgt_1),133).
orientation(lbgt_0,175,134). coords(lbgt_0,179,139,134).  visible(lbgt_0,134). happensAt(walking(lbgt_0),134). orientation(lbgt_1,155,134). coords(lbgt_1,76,88,134).  visible(lbgt_1,134). happensAt(walking(lbgt_1),134).
orientation(lbgt_0,175,135). coords(lbgt_0,182,141,135).  visible(lbgt_0,135). happensAt(walking(lbgt_0),135). orientation(lbgt_1,147,135). coords(lbgt_1,76,88,135).  visible(lbgt_1,135). happensAt(walking(lbgt_1),135).
orientation(lbgt_0,168,136). coords(lbgt_0,183,145,136).  visible(lbgt_0,136). happensAt(walking(lbgt_0),136). orientation(lbgt_1,147,136). coords(lbgt_1,77,88,136).  visible(lbgt_1,136). happensAt(walking(lbgt_1),136).
orientation(lbgt_0,168,137). coords(lbgt_0,185,148,137).  visible(lbgt_0,137). happensAt(walking(lbgt_0),137). orientation(lbgt_1,147,137). coords(lbgt_1,77,89,137).  visible(lbgt_1,137). happensAt(walking(lbgt_1),137).
orientation(lbgt_0,168,138). coords(lbgt_0,186,150,138).  visible(lbgt_0,138). happensAt(walking(lbgt_0),138). orientation(lbgt_1,147,138). coords(lbgt_1,77,89,138).  visible(lbgt_1,138). happensAt(walking(lbgt_1),138).
orientation(lbgt_0,168,139). coords(lbgt_0,188,152,139).  visible(lbgt_0,139). happensAt(walking(lbgt_0),139). orientation(lbgt_1,147,139). coords(lbgt_1,77,90,139).  visible(lbgt_1,139). happensAt(walking(lbgt_1),139).
orientation(lbgt_0,168,140). coords(lbgt_0,188,155,140).  visible(lbgt_0,140). happensAt(walking(lbgt_0),140). orientation(lbgt_1,153,140). coords(lbgt_1,77,89,140).  visible(lbgt_1,140). happensAt(walking(lbgt_1),140).
orientation(lbgt_0,168,141). coords(lbgt_0,189,157,141).  visible(lbgt_0,141). happensAt(walking(lbgt_0),141). orientation(lbgt_1,153,141). coords(lbgt_1,77,91,141).  visible(lbgt_1,141). happensAt(walking(lbgt_1),141).
orientation(lbgt_0,163,142). coords(lbgt_0,190,161,142).  visible(lbgt_0,142). happensAt(walking(lbgt_0),142). orientation(lbgt_1,153,142). coords(lbgt_1,77,92,142).  visible(lbgt_1,142). happensAt(walking(lbgt_1),142).
orientation(lbgt_0,163,143). coords(lbgt_0,189,165,143).  visible(lbgt_0,143). happensAt(walking(lbgt_0),143). orientation(lbgt_1,153,143). coords(lbgt_1,77,91,143).  visible(lbgt_1,143). happensAt(walking(lbgt_1),143).
orientation(lbgt_0,154,144). coords(lbgt_0,190,170,144).  visible(lbgt_0,144). happensAt(walking(lbgt_0),144). orientation(lbgt_1,153,144). coords(lbgt_1,77,92,144).  visible(lbgt_1,144). happensAt(walking(lbgt_1),144).
#show holdsAt/2.