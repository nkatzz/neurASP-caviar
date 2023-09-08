
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

	orientation(ricgt_3,153,111). coords(ricgt_3,192,170,111).  visible(ricgt_3,111). happensAt(walking(ricgt_3),111). orientation(ricgt_4,172,111). coords(ricgt_4,26,134,111).  visible(ricgt_4,111). happensAt(walking(ricgt_4),111).
orientation(ricgt_3,151,112). coords(ricgt_3,192,170,112).  visible(ricgt_3,112). happensAt(walking(ricgt_3),112). orientation(ricgt_4,172,112). coords(ricgt_4,26,134,112).  visible(ricgt_4,112). happensAt(walking(ricgt_4),112).
orientation(ricgt_3,150,113). coords(ricgt_3,192,169,113).  visible(ricgt_3,113). happensAt(walking(ricgt_3),113). orientation(ricgt_4,172,113). coords(ricgt_4,26,134,113).  visible(ricgt_4,113). happensAt(walking(ricgt_4),113).
orientation(ricgt_3,150,114). coords(ricgt_3,192,168,114).  visible(ricgt_3,114). happensAt(walking(ricgt_3),114). orientation(ricgt_4,172,114). coords(ricgt_4,26,134,114).  visible(ricgt_4,114). happensAt(walking(ricgt_4),114).
orientation(ricgt_3,150,115). coords(ricgt_3,192,167,115).  visible(ricgt_3,115). happensAt(walking(ricgt_3),115). orientation(ricgt_4,172,115). coords(ricgt_4,26,134,115).  visible(ricgt_4,115). happensAt(walking(ricgt_4),115).
orientation(ricgt_3,150,116). coords(ricgt_3,191,167,116).  visible(ricgt_3,116). happensAt(walking(ricgt_3),116). orientation(ricgt_4,172,116). coords(ricgt_4,26,134,116).  visible(ricgt_4,116). happensAt(walking(ricgt_4),116).
orientation(ricgt_3,150,117). coords(ricgt_3,191,165,117).  visible(ricgt_3,117). happensAt(walking(ricgt_3),117). orientation(ricgt_4,172,117). coords(ricgt_4,26,134,117).  visible(ricgt_4,117). happensAt(walking(ricgt_4),117).
orientation(ricgt_3,148,118). coords(ricgt_3,191,165,118).  visible(ricgt_3,118). happensAt(walking(ricgt_3),118). orientation(ricgt_4,172,118). coords(ricgt_4,26,134,118).  visible(ricgt_4,118). happensAt(walking(ricgt_4),118).
orientation(ricgt_3,147,119). coords(ricgt_3,191,163,119).  visible(ricgt_3,119). happensAt(walking(ricgt_3),119). orientation(ricgt_4,172,119). coords(ricgt_4,26,134,119).  visible(ricgt_4,119). happensAt(walking(ricgt_4),119).
orientation(ricgt_3,148,120). coords(ricgt_3,191,162,120).  visible(ricgt_3,120). happensAt(walking(ricgt_3),120). orientation(ricgt_4,172,120). coords(ricgt_4,26,134,120).  visible(ricgt_4,120). happensAt(walking(ricgt_4),120).
orientation(ricgt_3,149,121). coords(ricgt_3,191,160,121).  visible(ricgt_3,121). happensAt(walking(ricgt_3),121). orientation(ricgt_4,172,121). coords(ricgt_4,26,134,121).  visible(ricgt_4,121). happensAt(walking(ricgt_4),121).
orientation(ricgt_3,151,122). coords(ricgt_3,190,159,122).  visible(ricgt_3,122). happensAt(walking(ricgt_3),122). orientation(ricgt_4,172,122). coords(ricgt_4,26,134,122).  visible(ricgt_4,122). happensAt(walking(ricgt_4),122).
orientation(ricgt_3,152,123). coords(ricgt_3,190,158,123).  visible(ricgt_3,123). happensAt(walking(ricgt_3),123). orientation(ricgt_4,172,123). coords(ricgt_4,26,134,123).  visible(ricgt_4,123). happensAt(walking(ricgt_4),123).
orientation(ricgt_3,153,124). coords(ricgt_3,189,156,124).  visible(ricgt_3,124). happensAt(walking(ricgt_3),124). orientation(ricgt_4,172,124). coords(ricgt_4,26,134,124).  visible(ricgt_4,124). happensAt(walking(ricgt_4),124).
orientation(ricgt_3,153,125). coords(ricgt_3,189,156,125).  visible(ricgt_3,125). happensAt(walking(ricgt_3),125). orientation(ricgt_4,172,125). coords(ricgt_4,26,134,125).  visible(ricgt_4,125). happensAt(walking(ricgt_4),125).
orientation(ricgt_3,153,126). coords(ricgt_3,189,156,126).  visible(ricgt_3,126). happensAt(walking(ricgt_3),126). orientation(ricgt_4,172,126). coords(ricgt_4,26,134,126).  visible(ricgt_4,126). happensAt(walking(ricgt_4),126).
orientation(ricgt_3,153,127). coords(ricgt_3,189,156,127).  visible(ricgt_3,127). happensAt(walking(ricgt_3),127). orientation(ricgt_4,172,127). coords(ricgt_4,26,134,127).  visible(ricgt_4,127). happensAt(walking(ricgt_4),127).
orientation(ricgt_3,154,128). coords(ricgt_3,188,156,128).  visible(ricgt_3,128). happensAt(walking(ricgt_3),128). orientation(ricgt_4,172,128). coords(ricgt_4,26,134,128).  visible(ricgt_4,128). happensAt(walking(ricgt_4),128).
orientation(ricgt_3,155,129). coords(ricgt_3,188,156,129).  visible(ricgt_3,129). happensAt(inactive(ricgt_3),129). orientation(ricgt_4,172,129). coords(ricgt_4,26,134,129).  visible(ricgt_4,129). happensAt(walking(ricgt_4),129).
orientation(ricgt_3,156,130). coords(ricgt_3,188,156,130).  visible(ricgt_3,130). happensAt(inactive(ricgt_3),130). orientation(ricgt_4,172,130). coords(ricgt_4,26,134,130).  visible(ricgt_4,130). happensAt(walking(ricgt_4),130).
orientation(ricgt_3,156,131). coords(ricgt_3,188,156,131).  visible(ricgt_3,131). happensAt(inactive(ricgt_3),131). orientation(ricgt_4,172,131). coords(ricgt_4,26,134,131).  visible(ricgt_4,131). happensAt(walking(ricgt_4),131).
orientation(ricgt_3,156,132). coords(ricgt_3,188,156,132).  visible(ricgt_3,132). happensAt(inactive(ricgt_3),132). orientation(ricgt_4,172,132). coords(ricgt_4,26,134,132).  visible(ricgt_4,132). happensAt(walking(ricgt_4),132).
orientation(ricgt_3,156,133). coords(ricgt_3,188,156,133).  visible(ricgt_3,133). happensAt(inactive(ricgt_3),133). orientation(ricgt_4,172,133). coords(ricgt_4,26,134,133).  visible(ricgt_4,133). happensAt(walking(ricgt_4),133).
orientation(ricgt_3,155,134). coords(ricgt_3,187,156,134).  visible(ricgt_3,134). happensAt(inactive(ricgt_3),134). orientation(ricgt_4,172,134). coords(ricgt_4,26,134,134).  visible(ricgt_4,134). happensAt(walking(ricgt_4),134).
orientation(ricgt_3,155,135). coords(ricgt_3,187,156,135).  visible(ricgt_3,135). happensAt(inactive(ricgt_3),135). orientation(ricgt_4,172,135). coords(ricgt_4,26,134,135).  visible(ricgt_4,135). happensAt(walking(ricgt_4),135).
#show holdsAt/2.