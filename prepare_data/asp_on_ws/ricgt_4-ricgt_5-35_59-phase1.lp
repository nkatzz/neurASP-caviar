
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

	orientation(ricgt_4,172,35). coords(ricgt_4,31,122,35).  visible(ricgt_4,35). happensAt(walking(ricgt_4),35). orientation(ricgt_5,0,35). coords(ricgt_5,30,117,35).  visible(ricgt_5,35). happensAt(walking(ricgt_5),35).
orientation(ricgt_4,172,36). coords(ricgt_4,31,122,36).  visible(ricgt_4,36). happensAt(walking(ricgt_4),36). orientation(ricgt_5,0,36). coords(ricgt_5,30,117,36).  visible(ricgt_5,36). happensAt(walking(ricgt_5),36).
orientation(ricgt_4,172,37). coords(ricgt_4,31,122,37).  visible(ricgt_4,37). happensAt(walking(ricgt_4),37). orientation(ricgt_5,0,37). coords(ricgt_5,30,117,37).  visible(ricgt_5,37). happensAt(walking(ricgt_5),37).
orientation(ricgt_4,172,38). coords(ricgt_4,31,123,38).  visible(ricgt_4,38). happensAt(walking(ricgt_4),38). orientation(ricgt_5,0,38). coords(ricgt_5,30,117,38).  visible(ricgt_5,38). happensAt(walking(ricgt_5),38).
orientation(ricgt_4,172,39). coords(ricgt_4,31,123,39).  visible(ricgt_4,39). happensAt(walking(ricgt_4),39). orientation(ricgt_5,0,39). coords(ricgt_5,30,117,39).  visible(ricgt_5,39). happensAt(walking(ricgt_5),39).
orientation(ricgt_4,172,40). coords(ricgt_4,31,123,40).  visible(ricgt_4,40). happensAt(walking(ricgt_4),40). orientation(ricgt_5,0,40). coords(ricgt_5,30,117,40).  visible(ricgt_5,40). happensAt(walking(ricgt_5),40).
orientation(ricgt_4,172,41). coords(ricgt_4,31,123,41).  visible(ricgt_4,41). happensAt(walking(ricgt_4),41). orientation(ricgt_5,0,41). coords(ricgt_5,30,117,41).  visible(ricgt_5,41). happensAt(walking(ricgt_5),41).
orientation(ricgt_4,172,42). coords(ricgt_4,31,123,42).  visible(ricgt_4,42). happensAt(walking(ricgt_4),42). orientation(ricgt_5,0,42). coords(ricgt_5,30,117,42).  visible(ricgt_5,42). happensAt(walking(ricgt_5),42).
orientation(ricgt_4,172,43). coords(ricgt_4,30,124,43).  visible(ricgt_4,43). happensAt(walking(ricgt_4),43). orientation(ricgt_5,0,43). coords(ricgt_5,30,117,43).  visible(ricgt_5,43). happensAt(walking(ricgt_5),43).
orientation(ricgt_4,172,44). coords(ricgt_4,30,125,44).  visible(ricgt_4,44). happensAt(walking(ricgt_4),44). orientation(ricgt_5,0,44). coords(ricgt_5,30,117,44).  visible(ricgt_5,44). happensAt(walking(ricgt_5),44).
orientation(ricgt_4,172,45). coords(ricgt_4,30,125,45).  visible(ricgt_4,45). happensAt(walking(ricgt_4),45). orientation(ricgt_5,0,45). coords(ricgt_5,30,119,45).  visible(ricgt_5,45). happensAt(walking(ricgt_5),45).
orientation(ricgt_4,172,46). coords(ricgt_4,30,125,46).  visible(ricgt_4,46). happensAt(walking(ricgt_4),46). orientation(ricgt_5,0,46). coords(ricgt_5,30,119,46).  visible(ricgt_5,46). happensAt(walking(ricgt_5),46).
orientation(ricgt_4,172,47). coords(ricgt_4,30,125,47).  visible(ricgt_4,47). happensAt(walking(ricgt_4),47). orientation(ricgt_5,0,47). coords(ricgt_5,30,119,47).  visible(ricgt_5,47). happensAt(walking(ricgt_5),47).
orientation(ricgt_4,172,48). coords(ricgt_4,30,125,48).  visible(ricgt_4,48). happensAt(walking(ricgt_4),48). orientation(ricgt_5,0,48). coords(ricgt_5,30,119,48).  visible(ricgt_5,48). happensAt(walking(ricgt_5),48).
orientation(ricgt_4,172,49). coords(ricgt_4,30,125,49).  visible(ricgt_4,49). happensAt(walking(ricgt_4),49). orientation(ricgt_5,0,49). coords(ricgt_5,30,119,49).  visible(ricgt_5,49). happensAt(walking(ricgt_5),49).
orientation(ricgt_4,172,50). coords(ricgt_4,29,126,50).  visible(ricgt_4,50). happensAt(walking(ricgt_4),50). orientation(ricgt_5,0,50). coords(ricgt_5,30,119,50).  visible(ricgt_5,50). happensAt(walking(ricgt_5),50).
orientation(ricgt_4,172,51). coords(ricgt_4,29,127,51).  visible(ricgt_4,51). happensAt(walking(ricgt_4),51). orientation(ricgt_5,0,51). coords(ricgt_5,30,119,51).  visible(ricgt_5,51). happensAt(walking(ricgt_5),51).
orientation(ricgt_4,172,52). coords(ricgt_4,29,128,52).  visible(ricgt_4,52). happensAt(walking(ricgt_4),52). orientation(ricgt_5,0,52). coords(ricgt_5,30,119,52).  visible(ricgt_5,52). happensAt(walking(ricgt_5),52).
orientation(ricgt_4,172,53). coords(ricgt_4,29,128,53).  visible(ricgt_4,53). happensAt(walking(ricgt_4),53). orientation(ricgt_5,0,53). coords(ricgt_5,30,119,53).  visible(ricgt_5,53). happensAt(walking(ricgt_5),53).
orientation(ricgt_4,172,54). coords(ricgt_4,29,128,54).  visible(ricgt_4,54). happensAt(walking(ricgt_4),54). orientation(ricgt_5,0,54). coords(ricgt_5,29,120,54).  visible(ricgt_5,54). happensAt(walking(ricgt_5),54).
orientation(ricgt_4,172,55). coords(ricgt_4,29,128,55).  visible(ricgt_4,55). happensAt(walking(ricgt_4),55). orientation(ricgt_5,0,55). coords(ricgt_5,29,120,55).  visible(ricgt_5,55). happensAt(walking(ricgt_5),55).
orientation(ricgt_4,172,56). coords(ricgt_4,29,129,56).  visible(ricgt_4,56). happensAt(walking(ricgt_4),56). orientation(ricgt_5,0,56). coords(ricgt_5,29,121,56).  visible(ricgt_5,56). happensAt(walking(ricgt_5),56).
orientation(ricgt_4,172,57). coords(ricgt_4,29,129,57).  visible(ricgt_4,57). happensAt(walking(ricgt_4),57). orientation(ricgt_5,0,57). coords(ricgt_5,29,121,57).  visible(ricgt_5,57). happensAt(walking(ricgt_5),57).
orientation(ricgt_4,172,58). coords(ricgt_4,29,130,58).  visible(ricgt_4,58). happensAt(walking(ricgt_4),58). orientation(ricgt_5,0,58). coords(ricgt_5,29,121,58).  visible(ricgt_5,58). happensAt(walking(ricgt_5),58).
orientation(ricgt_4,172,59). coords(ricgt_4,28,130,59).  visible(ricgt_4,59). happensAt(walking(ricgt_4),59). orientation(ricgt_5,0,59). coords(ricgt_5,28,122,59).  visible(ricgt_5,59). happensAt(walking(ricgt_5),59).
#show holdsAt/2.