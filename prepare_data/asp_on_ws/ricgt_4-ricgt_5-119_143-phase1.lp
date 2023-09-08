
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

	orientation(ricgt_4,172,119). coords(ricgt_4,26,134,119).  visible(ricgt_4,119). happensAt(walking(ricgt_4),119). orientation(ricgt_5,0,119). coords(ricgt_5,27,127,119).  visible(ricgt_5,119). happensAt(walking(ricgt_5),119).
orientation(ricgt_4,172,120). coords(ricgt_4,26,134,120).  visible(ricgt_4,120). happensAt(walking(ricgt_4),120). orientation(ricgt_5,0,120). coords(ricgt_5,27,127,120).  visible(ricgt_5,120). happensAt(walking(ricgt_5),120).
orientation(ricgt_4,172,121). coords(ricgt_4,26,134,121).  visible(ricgt_4,121). happensAt(walking(ricgt_4),121). orientation(ricgt_5,0,121). coords(ricgt_5,27,127,121).  visible(ricgt_5,121). happensAt(walking(ricgt_5),121).
orientation(ricgt_4,172,122). coords(ricgt_4,26,134,122).  visible(ricgt_4,122). happensAt(walking(ricgt_4),122). orientation(ricgt_5,0,122). coords(ricgt_5,27,127,122).  visible(ricgt_5,122). happensAt(walking(ricgt_5),122).
orientation(ricgt_4,172,123). coords(ricgt_4,26,134,123).  visible(ricgt_4,123). happensAt(walking(ricgt_4),123). orientation(ricgt_5,0,123). coords(ricgt_5,27,127,123).  visible(ricgt_5,123). happensAt(walking(ricgt_5),123).
orientation(ricgt_4,172,124). coords(ricgt_4,26,134,124).  visible(ricgt_4,124). happensAt(walking(ricgt_4),124). orientation(ricgt_5,0,124). coords(ricgt_5,27,127,124).  visible(ricgt_5,124). happensAt(walking(ricgt_5),124).
orientation(ricgt_4,172,125). coords(ricgt_4,26,134,125).  visible(ricgt_4,125). happensAt(walking(ricgt_4),125). orientation(ricgt_5,0,125). coords(ricgt_5,27,127,125).  visible(ricgt_5,125). happensAt(walking(ricgt_5),125).
orientation(ricgt_4,172,126). coords(ricgt_4,26,134,126).  visible(ricgt_4,126). happensAt(walking(ricgt_4),126). orientation(ricgt_5,0,126). coords(ricgt_5,27,127,126).  visible(ricgt_5,126). happensAt(walking(ricgt_5),126).
orientation(ricgt_4,172,127). coords(ricgt_4,26,134,127).  visible(ricgt_4,127). happensAt(walking(ricgt_4),127). orientation(ricgt_5,0,127). coords(ricgt_5,27,127,127).  visible(ricgt_5,127). happensAt(walking(ricgt_5),127).
orientation(ricgt_4,172,128). coords(ricgt_4,26,134,128).  visible(ricgt_4,128). happensAt(walking(ricgt_4),128). orientation(ricgt_5,0,128). coords(ricgt_5,27,127,128).  visible(ricgt_5,128). happensAt(walking(ricgt_5),128).
orientation(ricgt_4,172,129). coords(ricgt_4,26,134,129).  visible(ricgt_4,129). happensAt(walking(ricgt_4),129). orientation(ricgt_5,0,129). coords(ricgt_5,27,127,129).  visible(ricgt_5,129). happensAt(walking(ricgt_5),129).
orientation(ricgt_4,172,130). coords(ricgt_4,26,134,130).  visible(ricgt_4,130). happensAt(walking(ricgt_4),130). orientation(ricgt_5,0,130). coords(ricgt_5,27,127,130).  visible(ricgt_5,130). happensAt(walking(ricgt_5),130).
orientation(ricgt_4,172,131). coords(ricgt_4,26,134,131).  visible(ricgt_4,131). happensAt(walking(ricgt_4),131). orientation(ricgt_5,0,131). coords(ricgt_5,27,127,131).  visible(ricgt_5,131). happensAt(walking(ricgt_5),131).
orientation(ricgt_4,172,132). coords(ricgt_4,26,134,132).  visible(ricgt_4,132). happensAt(walking(ricgt_4),132). orientation(ricgt_5,0,132). coords(ricgt_5,27,127,132).  visible(ricgt_5,132). happensAt(walking(ricgt_5),132).
orientation(ricgt_4,172,133). coords(ricgt_4,26,134,133).  visible(ricgt_4,133). happensAt(walking(ricgt_4),133). orientation(ricgt_5,0,133). coords(ricgt_5,27,127,133).  visible(ricgt_5,133). happensAt(walking(ricgt_5),133).
orientation(ricgt_4,172,134). coords(ricgt_4,26,134,134).  visible(ricgt_4,134). happensAt(walking(ricgt_4),134). orientation(ricgt_5,0,134). coords(ricgt_5,27,127,134).  visible(ricgt_5,134). happensAt(walking(ricgt_5),134).
orientation(ricgt_4,172,135). coords(ricgt_4,26,134,135).  visible(ricgt_4,135). happensAt(walking(ricgt_4),135). orientation(ricgt_5,0,135). coords(ricgt_5,27,127,135).  visible(ricgt_5,135). happensAt(walking(ricgt_5),135).
orientation(ricgt_4,172,136). coords(ricgt_4,26,135,136).  visible(ricgt_4,136). happensAt(walking(ricgt_4),136). orientation(ricgt_5,0,136). coords(ricgt_5,27,127,136).  visible(ricgt_5,136). happensAt(walking(ricgt_5),136).
orientation(ricgt_4,172,137). coords(ricgt_4,26,135,137).  visible(ricgt_4,137). happensAt(walking(ricgt_4),137). orientation(ricgt_5,0,137). coords(ricgt_5,27,127,137).  visible(ricgt_5,137). happensAt(walking(ricgt_5),137).
orientation(ricgt_4,172,138). coords(ricgt_4,26,135,138).  visible(ricgt_4,138). happensAt(walking(ricgt_4),138). orientation(ricgt_5,0,138). coords(ricgt_5,27,127,138).  visible(ricgt_5,138). happensAt(walking(ricgt_5),138).
orientation(ricgt_4,172,139). coords(ricgt_4,26,136,139).  visible(ricgt_4,139). happensAt(walking(ricgt_4),139). orientation(ricgt_5,0,139). coords(ricgt_5,27,127,139).  visible(ricgt_5,139). happensAt(walking(ricgt_5),139).
orientation(ricgt_4,172,140). coords(ricgt_4,26,136,140).  visible(ricgt_4,140). happensAt(walking(ricgt_4),140). orientation(ricgt_5,0,140). coords(ricgt_5,27,127,140).  visible(ricgt_5,140). happensAt(walking(ricgt_5),140).
orientation(ricgt_4,172,141). coords(ricgt_4,26,136,141).  visible(ricgt_4,141). happensAt(walking(ricgt_4),141). orientation(ricgt_5,0,141). coords(ricgt_5,27,127,141).  visible(ricgt_5,141). happensAt(walking(ricgt_5),141).
orientation(ricgt_4,172,142). coords(ricgt_4,26,136,142).  visible(ricgt_4,142). happensAt(walking(ricgt_4),142). orientation(ricgt_5,0,142). coords(ricgt_5,27,128,142).  visible(ricgt_5,142). happensAt(walking(ricgt_5),142).
orientation(ricgt_4,172,143). coords(ricgt_4,26,136,143).  visible(ricgt_4,143). happensAt(walking(ricgt_4),143). orientation(ricgt_5,0,143). coords(ricgt_5,27,128,143).  visible(ricgt_5,143). happensAt(walking(ricgt_5),143).
#show holdsAt/2.