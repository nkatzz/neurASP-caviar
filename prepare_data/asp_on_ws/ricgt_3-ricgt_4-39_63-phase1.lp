
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

	orientation(ricgt_3,51,39). coords(ricgt_3,301,286,39).  happensAt(appear(ricgt_3),39). happensAt(inactive(ricgt_3),39). orientation(ricgt_4,172,39). coords(ricgt_4,31,123,39).  visible(ricgt_4,39). happensAt(walking(ricgt_4),39).
orientation(ricgt_3,51,40). coords(ricgt_3,301,286,40).  visible(ricgt_3,40). happensAt(inactive(ricgt_3),40). orientation(ricgt_4,172,40). coords(ricgt_4,31,123,40).  visible(ricgt_4,40). happensAt(walking(ricgt_4),40).
orientation(ricgt_3,51,41). coords(ricgt_3,301,286,41).  visible(ricgt_3,41). happensAt(inactive(ricgt_3),41). orientation(ricgt_4,172,41). coords(ricgt_4,31,123,41).  visible(ricgt_4,41). happensAt(walking(ricgt_4),41).
orientation(ricgt_3,51,42). coords(ricgt_3,301,286,42).  visible(ricgt_3,42). happensAt(inactive(ricgt_3),42). orientation(ricgt_4,172,42). coords(ricgt_4,31,123,42).  visible(ricgt_4,42). happensAt(walking(ricgt_4),42).
orientation(ricgt_3,51,43). coords(ricgt_3,301,286,43).  visible(ricgt_3,43). happensAt(inactive(ricgt_3),43). orientation(ricgt_4,172,43). coords(ricgt_4,30,124,43).  visible(ricgt_4,43). happensAt(walking(ricgt_4),43).
orientation(ricgt_3,51,44). coords(ricgt_3,301,286,44).  visible(ricgt_3,44). happensAt(inactive(ricgt_3),44). orientation(ricgt_4,172,44). coords(ricgt_4,30,125,44).  visible(ricgt_4,44). happensAt(walking(ricgt_4),44).
orientation(ricgt_3,51,45). coords(ricgt_3,301,286,45).  visible(ricgt_3,45). happensAt(inactive(ricgt_3),45). orientation(ricgt_4,172,45). coords(ricgt_4,30,125,45).  visible(ricgt_4,45). happensAt(walking(ricgt_4),45).
orientation(ricgt_3,51,46). coords(ricgt_3,301,286,46).  visible(ricgt_3,46). happensAt(inactive(ricgt_3),46). orientation(ricgt_4,172,46). coords(ricgt_4,30,125,46).  visible(ricgt_4,46). happensAt(walking(ricgt_4),46).
orientation(ricgt_3,51,47). coords(ricgt_3,301,286,47).  visible(ricgt_3,47). happensAt(inactive(ricgt_3),47). orientation(ricgt_4,172,47). coords(ricgt_4,30,125,47).  visible(ricgt_4,47). happensAt(walking(ricgt_4),47).
orientation(ricgt_3,51,48). coords(ricgt_3,301,286,48).  visible(ricgt_3,48). happensAt(inactive(ricgt_3),48). orientation(ricgt_4,172,48). coords(ricgt_4,30,125,48).  visible(ricgt_4,48). happensAt(walking(ricgt_4),48).
orientation(ricgt_3,51,49). coords(ricgt_3,301,286,49).  visible(ricgt_3,49). happensAt(inactive(ricgt_3),49). orientation(ricgt_4,172,49). coords(ricgt_4,30,125,49).  visible(ricgt_4,49). happensAt(walking(ricgt_4),49).
orientation(ricgt_3,51,50). coords(ricgt_3,301,286,50).  visible(ricgt_3,50). happensAt(inactive(ricgt_3),50). orientation(ricgt_4,172,50). coords(ricgt_4,29,126,50).  visible(ricgt_4,50). happensAt(walking(ricgt_4),50).
orientation(ricgt_3,51,51). coords(ricgt_3,301,286,51).  visible(ricgt_3,51). happensAt(inactive(ricgt_3),51). orientation(ricgt_4,172,51). coords(ricgt_4,29,127,51).  visible(ricgt_4,51). happensAt(walking(ricgt_4),51).
orientation(ricgt_3,80,52). coords(ricgt_3,292,285,52).  visible(ricgt_3,52). happensAt(walking(ricgt_3),52). orientation(ricgt_4,172,52). coords(ricgt_4,29,128,52).  visible(ricgt_4,52). happensAt(walking(ricgt_4),52).
orientation(ricgt_3,80,53). coords(ricgt_3,290,283,53).  visible(ricgt_3,53). happensAt(walking(ricgt_3),53). orientation(ricgt_4,172,53). coords(ricgt_4,29,128,53).  visible(ricgt_4,53). happensAt(walking(ricgt_4),53).
orientation(ricgt_3,77,54). coords(ricgt_3,281,280,54).  visible(ricgt_3,54). happensAt(walking(ricgt_3),54). orientation(ricgt_4,172,54). coords(ricgt_4,29,128,54).  visible(ricgt_4,54). happensAt(walking(ricgt_4),54).
orientation(ricgt_3,70,55). coords(ricgt_3,280,279,55).  visible(ricgt_3,55). happensAt(walking(ricgt_3),55). orientation(ricgt_4,172,55). coords(ricgt_4,29,128,55).  visible(ricgt_4,55). happensAt(walking(ricgt_4),55).
orientation(ricgt_3,62,56). coords(ricgt_3,279,277,56).  visible(ricgt_3,56). happensAt(walking(ricgt_3),56). orientation(ricgt_4,172,56). coords(ricgt_4,29,129,56).  visible(ricgt_4,56). happensAt(walking(ricgt_4),56).
orientation(ricgt_3,58,57). coords(ricgt_3,278,277,57).  visible(ricgt_3,57). happensAt(walking(ricgt_3),57). orientation(ricgt_4,172,57). coords(ricgt_4,29,129,57).  visible(ricgt_4,57). happensAt(walking(ricgt_4),57).
orientation(ricgt_3,50,58). coords(ricgt_3,278,277,58).  visible(ricgt_3,58). happensAt(walking(ricgt_3),58). orientation(ricgt_4,172,58). coords(ricgt_4,29,130,58).  visible(ricgt_4,58). happensAt(walking(ricgt_4),58).
orientation(ricgt_3,50,59). coords(ricgt_3,277,276,59).  visible(ricgt_3,59). happensAt(walking(ricgt_3),59). orientation(ricgt_4,172,59). coords(ricgt_4,28,130,59).  visible(ricgt_4,59). happensAt(walking(ricgt_4),59).
orientation(ricgt_3,48,60). coords(ricgt_3,276,276,60).  visible(ricgt_3,60). happensAt(walking(ricgt_3),60). orientation(ricgt_4,172,60). coords(ricgt_4,28,130,60).  visible(ricgt_4,60). happensAt(walking(ricgt_4),60).
orientation(ricgt_3,45,61). coords(ricgt_3,272,276,61).  visible(ricgt_3,61). happensAt(walking(ricgt_3),61). orientation(ricgt_4,172,61). coords(ricgt_4,28,130,61).  visible(ricgt_4,61). happensAt(walking(ricgt_4),61).
orientation(ricgt_3,44,62). coords(ricgt_3,270,276,62).  visible(ricgt_3,62). happensAt(walking(ricgt_3),62). orientation(ricgt_4,172,62). coords(ricgt_4,28,130,62).  visible(ricgt_4,62). happensAt(walking(ricgt_4),62).
orientation(ricgt_3,40,63). coords(ricgt_3,265,276,63).  visible(ricgt_3,63). happensAt(walking(ricgt_3),63). orientation(ricgt_4,172,63). coords(ricgt_4,28,130,63).  visible(ricgt_4,63). happensAt(walking(ricgt_4),63).
#show holdsAt/2.